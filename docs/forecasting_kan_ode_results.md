# Full latent KAN-ODE training and validation

Experiment date: **2026-10-18**. This extends the completed
[implementation and smoke check](forecasting_kan_ode_implementation.md) with
the requested fixed three-seed training experiment. The research sequence
remains **KAN → validated future forecasting → KAN-ODE → conformal prediction**.

**Finding:** validation RMSE is **3.8381 ± 0.0200** across initialization seeds.
This is higher than the existing recurrent control's **3.8264 ± 0.0125** and
Ridge B's **3.7915**. This experiment does **not** demonstrate improved average
forecasting accuracy from the latent ODE. All runs and numerical checks
completed successfully, with no settings changed in response to results.

## Frozen task and architecture

Every initialization uses all **5,111 saved training samples from 2,025
participant prefixes**, and the same **851 validation samples from 338
prefixes**. The original assignments and training-fitted preprocessing are
reused without fitting or transformation again. All eligible quarters from a
participant remain in its existing partition. Source/input hashes and sample
alignment are checked against the original saved bundle.

The inputs remain `[N,2,48]` wearable packets in **SM1 then SM2** order and
`[N,63]` static context: enrollment demographics plus that quarter's measured
SM0 PHQ-9. The unchanged output is **measured SM3 PHQ-9 severity**, in original
0–27 questionnaire units, with an unbounded scalar estimate. There is no
prediction clipping, rounding, PHQ-change target, generated PHQ label, or SM3
sensing input.

The static KAN initializes a 16-value memory. One shared autonomous KAN
evolves it over relative coordinates `0→1`, `1→2`, and `2→3`. SM1 and SM2
update it through the same packet KAN and conventional GRUCell. The final
KAN head estimates SM3 severity. The **26,257-parameter** architecture,
fixed-grid cubic B-splines plus SiLU, dynamics initialization (base/spline
weights ×0.01, zero bias), and **RK4 step 0.25** are unchanged from the smoke
implementation. No settings respond to its preliminary RMSE.

This is an adaptation of Koenig, Kim, and Deng's
[*KAN-ODEs: Kolmogorov-Arnold Network Ordinary Differential Equations for
Learning Dynamical Systems and Hidden Physics*](https://arxiv.org/abs/2407.04192v3),
particularly `du/dt=KAN(u;θ)` in Eq. 7. Learned latent states, GRU observation
updates, endpoint-only PHQ supervision, B-splines, local RK4, and direct
PyTorch autograd are the disclosed adaptations in the implementation report.
SiLU and the paper's specified Swish are the same activation function.

Timing stays **conditional**: release/export indexing, no-look-ahead historical
summaries available by SM2, earlier measured SM0 availability, and enrollment
demographics without later replacement remain assumptions. SM2→SM3 means
**one relative study-month interval**, not 30 days. Exact available-at dates
and overlap with the target's two-week recall period are unverified. Only two
wearable packets are observed; solver evaluations are internal numerical
states, not additional observations or clinical trajectory measurements.

## Fixed training and actual control comparison

Before training, all three saved recurrent-control `training_config.json`
files and their provenance hashes were checked. Their actual
`gradient_clip` is **null**, meaning **no gradient clipping**. The new ODE
experiment uses the requested norm clipping at **5**. This is an optimization
difference, so the comparison is not a perfectly matched test of dynamics.

| Setting | This latent KAN-ODE | Existing non-ODE recurrent KAN |
|---|---|---|
| Initialization seeds | 42, 43, 44, fresh models | 42, 43, 44 |
| Optimizer | AdamW | AdamW |
| Learning rate / weight decay | 0.001 / 0.0001 | Same |
| Batch size / loss | 128 / endpoint MSE | Same |
| Maximum epochs / patience | 50 / 10 | Same |
| Selection / minimum improvement | Minimum validation RMSE / 0 | Same |
| Gradient norm clipping | **5** | **None** |
| Parameters | **26,257** | **23,169** |
| Between-observation dynamics | Shared autonomous KAN, three RK4 intervals | None |
| Device / precision / threads | CPU / float32 / 2 | Same |
| Minibatch-order seed | 42 for every initialization | Same |

Every seed starts from a newly constructed model after resetting its RNGs.
Initial checkpoints are saved and independently reconstructed from each seed;
no smoke, overfit, recurrent, or preceding-seed weights are resumed. Inherited
components retain their original same-seed initialization; only the appended
dynamics layer receives its documented initialization scaling.

The unchanged trainer retains the final partial minibatch, selects the lowest
validation RMSE, and restores that checkpoint before final evaluation. There
are **40 optimizer updates and seven validation batches per completed epoch**.
There is no architecture/hyperparameter search, learning-rate schedule, extra
training-only overfit run, or subsequent adjustment of solver settings.

## Validation results and aligned comparisons

| Initialization | MAE | RMSE | R² | Best epoch | Stopped epoch | Fit wall time |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2.838542 | 3.853956 | 0.611709 | 15 | 25 | 82.550 s |
| 43 | 2.878655 | 3.844812 | 0.613549 | 6 | 16 | 55.703 s |
| 44 | 2.867292 | 3.815639 | 0.619391 | 11 | 21 | 60.713 s |
| **Mean ± sample SD** | **2.861496 ± 0.020675** | **3.838136 ± 0.020012** | **0.614883 ± 0.004011** | — | 62 total epochs | **198.966 s total** |

All seeds stopped after ten epochs without improvement, with the selected
checkpoint restored. Fit wall time includes training, epoch validation,
gradient instrumentation/evidence writing, checkpoint/history writing, and
restoration. It excludes initial/restored solver diagnostics, independent
verification, data loading, and plotting. These local CPU times are not
hardware benchmarks or matched timing comparisons with previous models.

| Existing aligned comparison | MAE | RMSE | R² |
|---|---:|---:|---:|
| Persistence | 2.8496 | 4.0352 | 0.5743 |
| Ridge B: SM0 + demographics | **2.8087** | **3.7915** | **0.6242** |
| Ridge D: all allowed inputs | 2.8209 | 3.8136 | 0.6198 |
| Tabular KAN, mean ± sample SD | 2.9576 ± 0.0075 | 3.9120 ± 0.0038 | 0.5999 ± 0.0008 |
| Non-ODE recurrent KAN, mean ± sample SD | 2.8517 ± 0.0269 | 3.8264 ± 0.0125 | 0.6172 ± 0.0025 |
| **Latent KAN-ODE, mean ± sample SD** | **2.8615 ± 0.0207** | **3.8381 ± 0.0200** | **0.6149 ± 0.0040** |

MAE/RMSE use original PHQ-9 points; R² is dimensionless. Mean ± sample SD
uses `ddof=1` across three initializations on this one fixed split. It is
**not population uncertainty**, a confidence interval, or evidence of statistical
significance. No winning seed is selected for an ensemble or further tuning.
All comparison prediction files are checked against the exact same validation
sample indices and measured SM3 targets. Previous models are neither refitted
nor retuned.

## Numerical, gradient, and restoration diagnostics

Parameter hooks observe immutable gradient norms during backward, before
the original trainer clips them. Float64 norm calculation is diagnostic only
and does not change gradients or solver operations. Per-update global and
component norms, dynamics update sizes, and aggregate clipping counts are
saved. Post-update checks reject nonfinite parameters immediately.

All **16 trainable parameter tensors**, including all three dynamics tensors,
received finite/nonzero gradients in each run. Dynamics changed at every
optimizer update and at every restored checkpoint relative to initialization.
There were **zero numerical failures** and **no material step-halving warning**.

| Seed | Optimizer updates | Pre-clipping norm min / mean / max | Updates above 5 | Fit RHS evaluations |
|---:|---:|---|---:|---:|
| 42 | 1,000 | 9.730 / 88.098 / 324.934 | 1,000 / 1,000 | 56,400 |
| 43 | 640 | 9.184 / 87.918 / 416.581 | 640 / 640 | 36,096 |
| 44 | 840 | 11.367 / 74.774 / 318.074 | 840 / 840 | 47,376 |
| **Total** | **2,480** | — | **2,480 / 2,480** | **139,872** |

Thus clipping was active at **every** update; the difference from the unclipped
recurrent control is operational, not just a configuration label. Independent
first-minibatch backward calculations, with no optimizer step, reproduce the
saved pre-clipping global norms exactly for all seeds.

| Seed | Restored dynamics base-weight L2 change | Spline-weight L2 change | Bias L2 change |
|---:|---:|---:|---:|
| 42 | 0.763830 | 1.777487 | 0.090133 |
| 43 | 0.800962 | 1.669329 | 0.111232 |
| 44 | 0.683835 | 2.094524 | 0.089980 |

| Restored seed, step 0.25 versus 0.125 | Maximum prediction difference | RMS prediction difference | Maximum final-hidden difference |
|---:|---:|---:|---:|
| 42 | 0.000101089 | 0.000042426 | 0.000097752 |
| 43 | 0.000293732 | 0.000122393 | 0.000221252 |
| 44 | 0.000161171 | 0.000075195 | 0.000115395 |

The largest diagnostic prediction difference is below **0.000294 PHQ points**.
Initial-model differences are also finite and below threshold. Diagnostic
final-hidden magnitudes at step 0.25 reach 8.18, 11.48, and 9.05 respectively,
so some latent coordinates leave the spline core grid `[-3,3]`; the existing
SiLU base pathway extrapolates there. No grid adaptation or state clipping is
introduced. Small step-halving differences on this subset do not establish
that such extrapolation is clinically meaningful or globally accurate.

All dynamics parameter tensors must receive finite/nonzero gradients, change
during optimization, and differ from their fresh initialization at the selected
checkpoint. Recorded solver work is vectorized RHS calls, not a count of people
or measurements. At step 0.25 each batch requires three integrations, twelve
RK4 steps, and 48 RHS evaluations; a completed epoch therefore uses **141
integrations, 564 RK4 steps, and 2,256 RHS evaluations**, including validation.

The restored checkpoints are compared at steps **0.25 and 0.125 on the same
eight fixed training samples used in the implementation smoke diagnostic**,
with identical indices across all seeds. Both initial and restored diagnostics
are saved. Model weights and default step remain unchanged. No validation
prediction accuracy is used to select a step size. The pre-existing
**0.01 PHQ-point maximum prediction discrepancy** threshold is an engineering
stop rule, not a clinical tolerance or rigorous global error bound.

Nonfinite values or a material diagnostic warning stop execution before
performance interpretation. The runner preserves initial/best checkpoints,
incremental histories/gradient evidence, a failed-state checkpoint and failure
metadata; it never silently changes steps or clips predictions. Synthetic
failure-injection tests verify this behavior. Passing diagnostics on eight
inputs does not establish solver accuracy everywhere or scientifically correct
latent dynamics.

Every selected checkpoint is loaded into a fresh model and must reproduce
saved predictions and metrics **exactly**, with unchanged sample/target order.
Independent checks also verify fresh initialization, dynamics scaling,
early-stopping decisions, actual pre-clipping norms, and computational SM1
isolation from SM2. There is no carried state between samples or forward calls.

## Artifacts, commands, verification, and commits

New experiment code is `experiments/forecasting_kan_ode_full.py` and observational
gradient instrumentation is `training/gradient_diagnostics.py`, under
`src/kan_phenotyping/`. Existing model, solver, trainer, preprocessing, strict
data guards, retrospective workflow, and all earlier results are preserved.

Using the configured Python environment and a **fresh** output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/workspace/.cache/matplotlib XDG_CACHE_HOME=/workspace/.cache \
python -m kan_phenotyping.experiments.forecasting_kan_ode_full \
  --dataset-dir artifacts/conditional_forecasting/sm2-seed42 \
  --baseline-dir artifacts/conditional_forecasting_baselines/sm2-seed42 \
  --ablation-dir artifacts/conditional_forecasting_feature_ablations/sm2-seed42 \
  --tabular-dir artifacts/conditional_forecasting_kan/sm2-fixed-seeds42-44 \
  --recurrent-dir artifacts/conditional_forecasting_sequential_kan/sm2-fixed-seeds42-44 \
  --output artifacts/conditional_forecasting_kan_ode/sm2-fixed-seeds42-44
python -m pytest -q
```

The ignored output directory contains exact configurations/feature names,
original-input and consumed-artifact hashes, source hashes and software
versions, saved split/diagnostic indices, the unchanged preprocessor copy,
comparison provenance, and learning curves. Each seed has fresh/selected
checkpoints, training/validation histories, epoch runtimes and solver counts,
pre-clipping gradient records, dynamics-change evidence, initial/restored
step diagnostics, and aligned validation predictions. Original raw hashes are
inherited; raw data are not reopened. Calibration/test payloads are neither
opened nor hashed.

**All 422 repository tests passed**, including 46 new tests for observational
gradient instrumentation and the full experiment protocol. Tiny synthetic
fits test deterministic fresh initialization, unchanged preprocessing,
correct updates/solver counts, exact restoration, and failure preservation;
they do not add real-data training experiments.

**Twenty-six independent actual-run checks passed**, including fresh-process
direct-model restoration of all **3 × 851** saved predictions and manual
metrics, fresh initialization/scaling, early-stopping replay, source/input
hashes, solver counts, and gradient evidence. Saved predictions and metrics
are reproduced exactly for every seed. Python file-open guards recorded zero
raw/calibration/test payload attempts in training and independent verification;
this instrumentation is not an operating-system native-I/O sandbox.

All **106 previously tracked files**, **12 preceding uncommitted source/report/
test files**, and **132 earlier artifacts** remained unchanged. Four reserved
payload files were checked with metadata only, without reading or hashing
their contents. Independent scripts/evidence remain in
`/workspace/scratch/forecasting_kan_ode_full_validation/`.

Aggregate snapshots and plots are retained under
[`results/conditional_forecasting_extensions/`](../results/conditional_forecasting_extensions/README.md).
Participant identities, split arrays, individual targets/predictions, fitted
preprocessors, and model checkpoints stay in ignored local artifact storage.
As requested in the final instruction, the preceding uncommitted recurrent
work, ODE implementation/smoke work, and this full experiment are committed
separately after validation. Nothing is pushed.

## Interpretation and limits

Average RMSE is about **0.0117 points higher** than the recurrent control,
**0.0245 higher** than full Ridge D, and **0.0466 higher** than Ridge B. It is
lower than persistence and the earlier tabular KAN. Mean MAE is slightly
higher than persistence, so the improvements are not uniform across metrics.
Seed 44 is marginally below the same-seed recurrent RMSE, but that does not
reverse the average result or establish a reliable improvement. No subsequent
settings were changed and no seed was promoted as a winner.

The extra 3,088 parameters and gradient-clipping difference prevent attributing
any difference from the recurrent control solely to continuous-time evolution.
There is no MLP-ODE control, matched-capacity control, clipping ablation, or
KAN-specific-benefit test in this step. No claim is made that a gain recovers
clinical dynamics, establishes causality, or demonstrates methodological
novelty. The latent path is learned under endpoint supervision rather than
observed between packets.

Repeated quarters within participants are dependent, and pooled validation
metrics weight participants with more eligible quarters more heavily.
Complete-history selection and the original release's future-endpoint coverage
filter limit generalizability. Repeated validation checkpoint/model comparisons
create optimism; initialization spread does not address those effects. Test
generalization, prospective clinical validity, operational feature availability,
and a wholly future symptom interval remain unverified.

This step stops after the fixed training/validation experiment and the
requested local commits. No MLP-ODE, conformal prediction, calibration/test
evaluation, or push is performed.
