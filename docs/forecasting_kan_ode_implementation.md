# Paper-informed latent KAN-ODE: implementation and smoke validation

Implemented on **2026-10-18**. This step adds reusable dynamics and a solver,
checks their computation, and runs one limited smoke experiment. It does not
establish forecasting performance. The research direction remains
**KAN → validated future forecasting → KAN-ODE → conformal prediction**.

In simple terms, the recurrent control already maintains a 16-value memory
of a person's context and two wearable summaries. This extension lets a KAN
change that memory between observations. SM1 and SM2 each update the memory
through the existing GRU. After SM2, only the learned dynamics evolve the
memory toward the SM3 questionnaire estimate; no future sensing is supplied.
The memory coordinates are learned features, not measured clinical states.

**Outcome:** numerical and architecture checks passed, the separate overfit
check passed, and a fresh two-epoch smoke run completed without numerical
failures. Its validation RMSE was **9.2908 PHQ-9 points**. This is a short
implementation diagnostic, with no demonstrated predictive improvement.

## Attribution and disclosed adaptations

Benjamin C. **Koenig**, Suyong **Kim**, and Sili **Deng**,
*KAN-ODEs: Kolmogorov-Arnold Network Ordinary Differential Equations for
Learning Dynamical Systems and Hidden Physics*,
[arXiv:2407.04192v3](https://arxiv.org/abs/2407.04192v3),
[versioned PDF](https://arxiv.org/pdf/2407.04192v3).
References below use physical PDF pages, which match its printed page numbers.

| Paper location | Supporting formulation or passage | Implementation here |
|---|---|---|
| §IIA, pp. 3–4, Eqs. 4–5 | The study “adopts the more computationally efficient Gaussian radial basis functions (RBFs)”; Eq. 4 combines basis terms and a base pathway, with Gaussian Eq. 5 | Preserve our existing fixed-grid cubic B-splines, eight intervals over `[-3,3]`; no silent basis replacement |
| p. 4, following Eq. 5 | Swish residual function `b(x)=x·sigmoid(x)`, layer inputs normalized to `[-1,1]` | Existing SiLU base pathway is mathematically the same activation; no new layer normalization or rescaling to `[-1,1]` is added |
| §IIB, p. 4, Eqs. 6–8 | Eq. 7: `du/dt = KAN(u(t), θ)`; Eq. 8 integrates from `u0` | Autonomous `dh/dt = KANLinear(16,16)(h)` over learned latent coordinates |
| p. 5, Eq. 9 | MSE averaged over observed solution times | MSE only against final measured SM3 severity; no intermediate PHQ or derivative labels |
| p. 5 | “the adjoint sensitivity method is adopted”; Julia ecosystem and, unless otherwise specified, Tsit5 and ADAM | PyTorch, local fixed-step classical RK4, direct autograd through its operations, AdamW |

The paper links its [official implementation](https://github.com/DENG-MIT/KAN-ODEs)
on p. 15. Paper passages were checked in the supplied PDF, especially §II and
Eqs. 4–9. Learned latent coordinates, context initialization, GRU assimilation,
and endpoint-only PHQ supervision are our disclosed adaptations. This is not
an exact replication; KAN-ODE itself is the cited authors' existing method.
SiLU versus this Swish is a naming difference, not a different activation.

## Frozen conditional data and target

The [dataset report](forecasting_dataset.md) and
[non-ODE recurrent report](forecasting_sequential_kan.md) define the unchanged
cohort, saved preprocessing, and participant assignments. The loader verifies
the bundle manifest, source/input hashes, original feature order, same-quarter
month positions, measured target validity, and participant split separation.
Only saved training/validation observations and shared metadata are opened.
No raw CSV is reopened, eligibility rebuilt, preprocessing refitted or
reapplied, or participant assignment changed.

| Payload | Samples | Participant prefixes | Wearables | Static context | Target |
|---|---:|---:|---|---|---|
| Original saved training | 5,111 | 2,025 | `[5111,2,48]` | `[5111,63]` | `[5111]` |
| Smoke training subset | **340** | **128** | `[340,2,48]` | `[340,63]` | `[340]` |
| Validation monitoring | **851** | **338** | `[851,2,48]` | `[851,63]` | `[851]` |

Packets are ordered **SM1 then SM2**, with 27 Statistics values and 21 saved
training-derived missingness indicators each. Static context contains 62
transformed enrollment demographic columns and one standardized, measured
SM0 score paired to that quarter. Reflattening exactly recovers the same 159
float32 values used previously. The output is raw **measured SM3 PHQ-9
severity**, not change or an intermediate generated PHQ label.

Smoke selection is outcome-independent: sort unique saved training participant
strings, permute with NumPy generator seed 42, take 128, and retain **every**
saved training sample for those participants in canonical order. This is a
subset for a computational smoke run, not a new participant partition.
The full saved training set remains the source for train-only preprocessing.

The four conditional assumptions remain: release/export index correspondence,
historical summaries available by SM2 without future backfill, measured SM0
available earlier, and enrollment demographics without later replacement.
Times `0,1,2,3` are relative study-month coordinates. The SM2-to-SM3 horizon is
**one relative study-month interval**, not 30 days. Exact available-at dates
and nonoverlap with the target's two-week recall period remain unverified.
No SM3 wearable/intervention input or undocumented regression summary enters.
Calibration/test predictor and outcome payloads remain unused.

## Architecture and initialization

For context `c`, historical packets `w1,w2`, and the same autonomous `f`:

```text
h0        = static_KAN(c)                         63 → 16
h1_before = integrate(f, h0,       0, 1)
h1_after  = shared_GRU(shared_packet_KAN(w1), h1_before)
h2_before = integrate(f, h1_after, 1, 2)
h2_after  = shared_GRU(shared_packet_KAN(w2), h2_before)
h3        = integrate(f, h2_after, 2, 3)
prediction = KAN_head(h3)                         16 → 1

f(h) = KANLinear(16,16)(h)
shared_packet_KAN: 48 → 16; shared GRUCell: input/hidden width 16
```

Static context initializes the state once. Each call starts from its own
context, with no carried hidden state between people, samples, batches, or
forward calls. The six-state trace exposes each pre/post-observation state.
Dynamics are an independent module passed to a parameter-free integrator, so
a future dynamics control can replace them. No additional model is implemented
or trained here.

| Component | Parameters |
|---|---:|
| Static KAN | 12,112 |
| Shared packet KAN | 9,232 |
| Shared conventional GRUCell | 1,632 |
| KAN head | 193 |
| Autonomous dynamics KAN | 3,088 |
| **Total** | **26,257** |

The inherited components are constructed in their original order before
creating dynamics. Their same-seed initialization is exactly unchanged.
Only newly initialized dynamics base/spline weights are multiplied by **0.01**,
with zero bias. Existing spline grids and SiLU pathways remain unchanged.
Output is an unbounded scalar: no sigmoid, prediction clipping, or rounding.

## Local solver and independent numerical checks

`numerics/rk4.py` provides `RK4Integrator`, implementation version **1.0**.
The active environment has PyTorch **2.9.1+cpu**; `torchdiffeq`, `torchode`, and
`torchdyn` were not installed. No new solver dependency was installed.
For a step `d`, classical RK4 computes:

```text
k1 = f(h)
k2 = f(h + d*k1/2)
k3 = f(h + d*k2/2)
k4 = f(h + d*k3)
h_next = h + d*(k1 + 2*k2 + 2*k3 + k4)/6
```

Default `d=0.25` in relative-month units, with an exact final partial interval
when needed. A zero-duration call returns the input tensor unchanged without
evaluating dynamics. Backward intervals are also tested. Integration uses
out-of-place PyTorch arithmetic, with no detachment or NumPy conversion;
autograd differentiates the actual discrete solver. Nonfinite stages,
derivatives, outputs, losses, and gradients fail explicitly. No automatic step
change or prediction clipping masks a failure. Parameter initialization uses
in-place scaling outside the integration path, before training.

Independent synthetic tests include:

- Constant trajectories including partial/backward final steps, accurate to
  `1e-14` in float64; two-state linear trajectories against `torch.matrix_exp`
  with `rtol=2e-7, atol=2e-8` at step 0.125.
- Approximately 16-fold error reduction on halving the step for an exponential
  solution, consistent with fourth-order convergence.
- Input and dynamics-parameter gradients for `h'=a*h`, checked against analytic
  derivatives and an independent centered finite difference (`epsilon=1e-6`,
  gradient tolerance `rtol=1e-8, atol=1e-9` for the finite-difference check).
- Zero dynamics exactly recover the copied non-ODE recurrent model, including
  its predictions. Zero-duration integration preserves identity and gradients.
- Shared encoders/GRU/dynamics, requested ordering, finite/nonzero gradients
  in every parameter tensor, checkpoint restoration, and no carried state.
- Changing SM2 leaves all states through `h1_after` bitwise unchanged;
  its gradient with respect to SM2 is exactly zero. A prefix computation
  receiving only context and SM1 reproduces that update.

Tiny batch-layout roundoff is allowed when comparing individually evaluated
samples to a whole batch; exact checkpoint and zero-dynamics checks remain
bitwise. These are computation checks, not proof of scientifically correct
or clinically interpretable latent dynamics.

## Separate overfit check and limited smoke run

The training-only diagnostic selects 32 saved training samples with seed 2026.
A separate model uses full-batch AdamW (`lr=0.001`, weight decay `0.0001`), MSE,
and gradient norm clipping at 5, with a fixed 1,000-step cap. Its predeclared
pass rule requires RMSE ≤0.25, MSE ≤1% of initial MSE, changed weights, and
finite/nonzero gradients in every parameter tensor.

| Overfit result | Value |
|---|---:|
| Initial MSE | 57.447731 |
| Final MSE / RMSE | 0.056493 / **0.237683** |
| Steps / optimization wall time | **100 / 4.350 s** |
| Finite/nonzero gradients | All 16 parameter tensors |
| Result | **Passed** |

Its model and optimizer are discarded. The smoke model starts fresh from
seed **42**; a saved initial checkpoint independently verifies this reset.
Smoke settings are fixed: **two epochs**, batch size 128, AdamW `lr=0.001`,
weight decay `0.0001`, MSE, gradient norm clipping **5**, CPU float32, two
PyTorch threads, deterministic algorithms, and minibatch-order seed 42.
There are three optimizer batches per epoch, retaining the partial batch.
The unchanged trainer selects/restores the lowest validation RMSE within
these two epochs; no extended training or tuning follows.

| Smoke epoch | Online training MSE | Validation MAE | Validation RMSE | Validation R² |
|---:|---:|---:|---:|---:|
| 1 | 90.105143 | 7.313938 | 9.516238 | −1.367417 |
| **2, restored** | **86.338071** | **7.082089** | **9.290824** | **−1.256590** |

Fit wall time was **0.804 seconds**, including per-epoch validation and
checkpoint writing/restoration; it excludes overfit, numerical diagnostics,
independent verification, and plotting. These are local wall times, not a
hardware benchmark. Six optimizer updates reached all 16 parameter tensors
with finite/nonzero gradients. There were **zero numerical failures**.
The negative R² and large errors make this an implementation smoke result,
not a useful fitted forecasting baseline or a fair comparison with prior
full-cohort, converged models.

At step 0.25, each model batch uses three integration calls, twelve RK4 steps,
and **48 RHS evaluations**, shared across samples in that batch. Fit plus
epoch validation used **60 integration calls, 240 steps, 960 RHS evaluations**
(six training and fourteen validation batches). The separate overfit check,
including its evaluations, used 603 integration calls and 9,648 RHS evaluations.
Counts refer to vectorized derivative calls, not individual observed packets.

The same eight fixed smoke-training inputs were checked at steps **0.25 and
0.125** before and after smoke training, using the fresh and restored models:

| Diagnostic | Maximum prediction difference | RMS prediction difference | Maximum final-hidden difference |
|---|---:|---:|---:|
| Before smoke | 1.49e−8 | 9.50e−9 | 8.94e−8 |
| After smoke | 1.19e−7 | 5.58e−8 | 1.79e−7 |

All states and predictions were finite, with no diagnostic instability flag
at the predeclared 0.01-PHQ-point discrepancy threshold. This threshold is an
engineering warning, not a clinical tolerance or rigorous global error bound.
The diagnostic leaves weights and default step 0.25 unchanged. No validation
accuracy is used to select a step. Internal RK4 states are numerical evaluation
points; they do not manufacture daily observations or intermediate PHQ labels.

## Reproduction, artifacts, and preservation

New reusable code is `models/latent_kan_ode.py`, `numerics/rk4.py`, and
`experiments/forecasting_kan_ode.py` under `src/kan_phenotyping/`. Existing
models, loaders, trainers, guards, retrospective results, and forecasting
results are preserved, including the preceding uncommitted recurrent work.

Run with the configured environment and a **fresh** ignored output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/workspace/.cache/matplotlib XDG_CACHE_HOME=/workspace/.cache \
python -m kan_phenotyping.experiments.forecasting_kan_ode \
  --dataset-dir artifacts/conditional_forecasting/sm2-seed42 \
  --output artifacts/conditional_forecasting_kan_ode/sm2-seed42-smoke
```

Artifacts in that directory include configuration/exact feature lists,
`run.json` with consumed-input, source, original-input and artifact hashes,
the unchanged preprocessor copy, private sample/subset definitions,
`sanity/` diagnostic weights/history, and `smoke/` initial/best checkpoints,
two-epoch history, aligned validation predictions, metrics, and both step-size
diagnostics. Learning curves are saved as PNG/SVG. None of these private
artifacts or identifiers is added to source control.

```bash
python -m pytest -q tests/test_rk4.py tests/test_latent_kan_ode.py \
  tests/test_forecasting_kan_ode.py
python -m pytest -q
```

Final test and independent verification evidence is recorded in the ignored
`validation_checks.json` and `access_checks.json`. The execution/verification
scripts reside outside source control at
`/workspace/scratch/forecasting_kan_ode_validation/`. Python open-audit guards
block raw and calibration/test payload reads; their scope is Python file-open
events, not an operating-system native-I/O sandbox.

**All 376 repository tests passed**, including **89 new tests**: 34 numerical
solver checks, 26 model checks, and 29 synthetic experiment-protocol checks.
The protocol checks exercise deterministic participant selection, exact
month/sample alignment, rejection of altered preprocessing or subset content,
fresh initialization, tiny reproducible two-epoch fits, restoration, and
failure reporting. They use synthetic data, not additional forecasting runs.
**Eighteen independent actual-run checks passed**, including fresh-process
restoration, manual metrics, unchanged preprocessing/input hashes, and
reproduction of the pre/post-training step diagnostics. Both execution and
verification guards recorded zero raw/calibration/test payload open attempts.
All **106 pre-existing tracked files**, **four preceding sequential files**,
and **111 earlier artifact files** were preserved; four reserved payloads were
checked with file metadata only, without opening or hashing their contents.

## Limits and stopping point

Only **two historical wearable packets** are observed. A continuous latent
path between them is imposed by the architecture and is not identified as
clinical physiology by an endpoint-only loss. Unverified exact timing,
questionnaire recall overlap, complete-history selection, original release
selection, and dependent repeated quarters remain limitations. Participant
separation does not remove those limitations. Repeated validation use can
also create optimism; calibration/test generalization remains unknown.

No clinical trajectory recovery, causal interpretation, predictive benefit,
methodological novelty, or prospective clinical validation is demonstrated.
The step ends after implementation, numerical checks, a separate overfit
check, and one two-epoch smoke run. No full three-seed experiment, dynamics
control training, tuning, conformal prediction, commit, or push was performed.
