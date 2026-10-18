# MLP-dynamics-only ODE control

Experiment date: **2026-10-18**. This step compares two latent derivative
parameterizations while retaining the existing KAN encoders, GRU observation
updates, KAN prediction head, and integration schedule. It follows the
[KAN-ODE implementation](forecasting_kan_ode_implementation.md),
[full KAN-ODE results](forecasting_kan_ode_results.md), and
[optimization-matched recurrent control](forecasting_matched_recurrent_control.md).
All previous workflows and results are preserved.

**Finding:** MLP-dynamics validation RMSE is **3.8382 ± 0.0041**, compared with
KAN dynamics **3.8381 ± 0.0200** and the matched non-ODE recurrent control
**3.8326 ± 0.0090**. MLP-minus-KAN mean RMSE differs by about **+0.000020**
PHQ points, with different per-seed results. Numerically close means do **not**
establish statistical equivalence. Neither ODE control beats Ridge B or D
on average RMSE in this comparison. Initialization SD is not population
uncertainty or a confidence interval.

In simple terms, the same learned memory still evolves between observations
and is updated by the same two wearable packets. Only the function that
changes that memory is replaced with a small conventional neural network.
This is **MLP latent dynamics inside the existing KAN-encoded architecture**,
not an entirely MLP-based model or a recovery of measured clinical dynamics.

## Fixed architecture and actual parameter counts

```text
h0        = static_KAN(context)                       63 → 16
h1_before = RK4(f, h0,       0, 1)
h1_after  = shared_GRU(shared_packet_KAN(SM1), h1_before)
h2_before = RK4(f, h1_after, 1, 2)
h2_after  = shared_GRU(shared_packet_KAN(SM2), h2_before)
h3        = RK4(f, h2_after, 2, 3)
prediction = KAN_head(h3)                             16 → 1

f_MLP = Linear(16,93) → Tanh → Linear(93,16)
f_KAN = KANLinear(16,16)          [previous comparison only]
```

The MLP model inherits the exact existing `trace_states` and `forward`
methods. Static context initializes the hidden state once; the packet encoder,
GRU, and derivative each share their parameters over their respective calls.
No sensing arrives after SM2. The unbounded scalar output has no sigmoid,
clipping, rounding, or generated intermediate PHQ labels.

| Component | Parameters |
|---|---:|
| Static KAN | 12,112 |
| Shared packet KAN | 9,232 |
| Shared GRUCell | 1,632 |
| KAN prediction head | 193 |
| **Unchanged shared components** | **23,169** |
| MLP first Linear: `16×93 + 93` | 1,581 |
| MLP final Linear: `93×16 + 16` | 1,504 |
| **MLP dynamics** | **3,085** |
| Existing KAN dynamics | 3,088 |
| **MLP-dynamics total** | **26,254** |
| Existing KAN-ODE total | 26,257 |

Counts were computed from actual model parameters. The dynamics modules are
approximately count-matched, differing by three parameters. Width **93** is
fixed as requested; no alternative width, architecture, or hyperparameter
search is run. Approximate parameter matching does not match function classes,
representational capacity, initialization distributions, or optimization paths.

## Frozen conditional data and optimization

All **5,111 training samples from 2,025 participant prefixes** and the same
**851 validation samples from 338 prefixes** are reused. Wearables remain
`[N,2,48]` ordered SM1 then SM2; context is `[N,63]`, consisting of 62
transformed enrollment demographic columns and the measured, standardized
SM0 score paired to each quarter. The target remains raw **measured SM3
PHQ-9 severity**, not change or a generated label.

Saved participant assignments, eligibility, preprocessing, missingness
indicators, and feature order are unchanged. No raw data are reopened, cohort
rebuilt, preprocessing fitted or reapplied, or target imputed. No SM3 sensing,
intervention inputs, or undocumented regression summaries are introduced.
Every reference prediction file is checked against exact validation indices
and measured targets; its metrics are independently recomputed.

The four conditional assumptions remain: release/export indexing correspondence,
historical summaries available by SM2 without future backfill, earlier measured
SM0 availability, and enrollment demographics without later replacement.
Coordinates `0,1,2,3` represent relative study months, not dates. The SM2→SM3
horizon is **one relative study-month interval**, not 30 days. Exact available-at
timestamps and overlap with the PHQ-9 two-week recall period remain unverified.
This is not prospective clinical validation.

| Setting | Fixed value, matching existing ODE/control experiments |
|---|---|
| Fresh initialization seeds | 42, 43, 44 |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 |
| Batch size / loss | 128 / endpoint MSE |
| Maximum epochs / patience | 50 / 10 |
| Checkpoint selection | Minimum validation RMSE, minimum improvement 0; restore before evaluation |
| Gradient norm clipping | 5; no clipping of predictions |
| Solver | Existing local classical RK4 v1.0, step 0.25, direct autograd |
| Compute | CPU, float32 model/state, two PyTorch threads, deterministic algorithms |
| Minibatch order | Generator seed 42 for all initialization seeds; zero workers; final partial batch retained |
| Scheduler / search | None / none |

Actual saved optimization configurations and input/source hashes are checked
against KAN-ODE and the matched recurrent control. Existing references are
read from aggregate artifacts and saved validation predictions; their trained
weights are not loaded and no reference model is refitted or retuned.

## Initialization and initial derivative magnitudes

For every seed, all **16 shared state tensors**—13 parameter tensors and three
spline-grid buffers—match the corresponding saved **initial** KAN-ODE artifact
exactly, including static encoder, packet encoder, GRU, and output head. Shared
components are constructed first, before the MLP consumes random draws. An
RNG-free, parameter-free constructor placeholder is replaced before any forward
call; a discarded random KAN dynamics layer is not constructed first.

The MLP uses standard PyTorch Linear initialization. Its final layer weights
are then multiplied by **0.01**, and its final bias is zeroed. First-layer
weights/bias retain their standard initialization. No weights are copied from
a trained checkpoint. Shared matching is verified before any full training,
and checked again on each fresh experiment model after the overfit diagnostic.

The prescribed MLP and KAN initialization rules are **not identical in function
space**. On the same eight fixed training examples used by the previous ODE
diagnostics, both initial derivative functions are evaluated at exactly the
same `h0 = static_KAN(context)`. No validation outcomes select or adjust their
initialization. Values below are coordinate RMS in learned hidden units per
relative-month coordinate, not PHQ derivatives or clinical measurements.

| Seed | Initial MLP coordinate RMS | Initial KAN coordinate RMS | RMS of derivative difference | MLP maximum absolute coordinate |
|---:|---:|---:|---:|---:|
| 42 | 0.001365412 | 0.001059337 | 0.001908554 | 0.003924796 |
| 43 | 0.001578724 | 0.001517236 | 0.002122104 | 0.004373298 |
| 44 | 0.001256200 | 0.001007767 | 0.001672932 | 0.004070485 |

Per-state L2 mean/maximum magnitudes are also saved in each seed's
`initial_derivatives.json`. Derivative comparisons use only initial KAN
dynamics and preserve model parameters and RNG state.

## Checks before full training and separate overfit diagnostic

Before training all three initialization seeds pass actual-data checks:

- Zeroing the MLP dynamics reproduces the shared-weight non-ODE recurrent
  control (`rtol=1e-6`, `atol=1e-6`; observed difference zero).
- All **17 parameter tensors**, including all four MLP tensors, receive finite,
  nonzero gradients from the measured endpoint loss.
- The integration schedule, shared observation encoders/GRU, and SM1→SM2 order
  remain unchanged. Changing SM2 cannot change the state after SM1; its gradient
  with respect to SM2 is exactly zero, and a prefix-only computation agrees.
- Hidden state does not carry between samples, batches, or forward calls.
- All **34 existing RK4 tests** still pass, including analytic constant/linear
  ODEs, partial/zero intervals, fourth-order convergence, independent gradient
  checks, and explicit failure on nonfinite states.

A separate seed-2026 model overfits **32 training samples only**, full batch,
AdamW 0.001/0.0001, MSE, clipping 5. The predeclared cap is 1,000 steps; success
requires RMSE ≤0.25, MSE ≤1% of initial MSE, and finite/nonzero gradients in
every parameter tensor. It has no validation input.

| Overfit check | Result |
|---|---:|
| Initial training MSE | 57.433681 |
| Final training MSE / RMSE | 0.057153 / **0.239066** |
| Steps / optimization time | **101 / 1.674 s** |
| Result | **Passed** |

Its weights/optimizer are discarded and saved separately under `sanity/`.
Every full run starts fresh from its specified seed; none resumes this model.
This diagnostic checks optimization, not generalization.

## Validation results and per-seed comparisons

| MLP-dynamics seed | MAE | RMSE | R² | Best epoch | Stopped epoch | Fit wall time |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2.867480 | 3.838771 | 0.614762 | 10 | 20 | 14.938 s |
| 43 | 2.869836 | 3.833769 | 0.615766 | 13 | 23 | 17.339 s |
| 44 | 2.838114 | 3.841928 | 0.614128 | 15 | 25 | 22.028 s |
| **Mean ± sample SD** | **2.858476 ± 0.017674** | **3.838156 ± 0.004114** | **0.614886 ± 0.000826** | — | 68 total epochs | **54.304 s total** |

All seeds stopped after ten epochs without improvement, with the selected
checkpoint restored. Fit wall times include training, epoch validation,
gradient instrumentation/evidence, history/checkpoint writing, and restoration.
They exclude overfit, initial/restored numerical checks, independent checking,
data loading, and plotting. These are local CPU times, not hardware benchmarks.

| Aligned validation reference | MAE | RMSE | R² |
|---|---:|---:|---:|
| Matched recurrent, clip 5 | 2.8645 ± 0.0152 | 3.8326 ± 0.0090 | 0.6160 ± 0.0018 |
| Existing KAN-ODE | 2.8615 ± 0.0207 | 3.8381 ± 0.0200 | 0.6149 ± 0.0040 |
| **MLP-dynamics ODE** | **2.8585 ± 0.0177** | **3.8382 ± 0.0041** | **0.6149 ± 0.0008** |
| Ridge B: SM0 + demographics | **2.8087** | **3.7915** | **0.6242** |
| Ridge D: all allowed inputs | 2.8209 | 3.8136 | 0.6198 |

The following differences are **MLP-dynamics ODE minus reference**. Negative
MAE/RMSE differences favor MLP; positive R² differences favor MLP. Ridge has
one deterministic saved result reused across rows, not three fitted seeds.

| Seed | Reference | Δ MAE | Δ RMSE | Δ R² |
|---:|---|---:|---:|---:|
| 42 | KAN-ODE | +0.028938 | −0.015186 | +0.003054 |
| 43 | KAN-ODE | −0.008819 | −0.011043 | +0.002217 |
| 44 | KAN-ODE | −0.029179 | +0.026289 | −0.005263 |
| 42 | Matched recurrent | −0.014558 | +0.001691 | −0.000339 |
| 43 | Matched recurrent | +0.014201 | −0.004631 | +0.000929 |
| 44 | Matched recurrent | −0.017826 | +0.019725 | −0.003952 |
| 42 | Ridge B | +0.058743 | +0.047256 | −0.009426 |
| 43 | Ridge B | +0.061099 | +0.042254 | −0.008423 |
| 44 | Ridge B | +0.029377 | +0.050414 | −0.010060 |
| 42 | Ridge D | +0.046550 | +0.025138 | −0.005029 |
| 43 | Ridge D | +0.048906 | +0.020136 | −0.004026 |
| 44 | Ridge D | +0.017184 | +0.028296 | −0.005663 |

MAE/RMSE are in original PHQ-9 points; R² is dimensionless. All models use
the identical 851 validation examples. Mean ± sample SD uses `ddof=1` across
initializations on one fixed participant split. It is not population
uncertainty, a confidence interval, statistical equivalence, or a significance
test. No seed is promoted, ensembled, or used for follow-up tuning.

## Numerical behavior, gradients, and solver evaluations

All 17 shared/dynamics parameter tensors receive finite/nonzero gradients.
All four selected MLP dynamics tensors change from initialization; their L2
changes and per-update changes are saved. Pre-clipping norms are observed
without changing the existing trainer's updates. All updated parameters,
predictions, states, and losses remain finite. There are **zero numerical
failures** and no material step-halving warning.

| Seed | Optimizer updates | Updates above clip 5 | Pre-clipping norm min / mean / max | Fit RHS evaluations |
|---:|---:|---:|---|---:|
| 42 | 800 | 800 | 7.711 / 69.674 / 262.775 | 45,120 |
| 43 | 920 | 920 | 5.266 / 64.523 / 244.238 | 51,888 |
| 44 | 1,000 | 1,000 | 6.805 / 66.416 / 365.181 | 56,400 |
| **Total** | **2,720** | **2,720** | — | **153,408** |

Clipping is active at every update. Each step-0.25 batch evaluates the shared
derivative 48 times: three intervals, twelve RK4 steps, four stages per step.
With 40 training and seven validation batches per epoch, fit totals are
**9,588 integration calls, 38,352 RK4 steps, and 153,408 vectorized RHS calls**.
These counts include epoch validation and exclude separate diagnostics/final
reproduction. Per-phase counters and learning curves are retained.

The same fixed eight training inputs are evaluated at **0.25 and 0.125** on
fresh and restored models. Neither validation accuracy nor the result below
selects solver settings; all training uses 0.25.

| Restored seed | Maximum prediction difference | RMS prediction difference | Maximum final-hidden difference |
|---:|---:|---:|---:|
| 42 | 0.000059843 | 0.000035167 | 0.000075817 |
| 43 | 0.000111341 | 0.000090895 | 0.000147104 |
| 44 | 0.000030041 | 0.000025394 | 0.000030041 |

All are below the existing **0.01 PHQ-point** engineering stop threshold.
This is a diagnostic on eight inputs, not a clinical tolerance or rigorous
global error bound. Initial checks are also finite/below threshold. Some
hidden values lie outside the shared KAN head's spline core; its existing
SiLU base pathway extrapolates without state/prediction clipping or grid
adaptation. Small solver differences do not establish meaningful physiology.

Nonfinite values or a material solver warning stop the experiment. Full-training
failures preserve failed-state checkpoints, histories, diagnostics, and failure metadata. No
silent step changes or prediction clipping mask failures. Synthetic tests
exercise this stop path. Restored models reproduce saved predictions/metrics
exactly, with no hidden-state carry or SM1 dependence on SM2.

## Artifacts, verification, and reproduction

New reusable files are `models/latent_mlp_ode.py` and
`experiments/forecasting_mlp_ode.py` under `src/kan_phenotyping/`, plus
`tests/test_forecasting_mlp_ode.py` and this report. Existing source, model,
solver, trainer, saved loaders, preprocessing, reports, and results are unchanged,
including the preceding uncommitted matched recurrent work.

The ignored run directory is
`artifacts/conditional_forecasting_mlp_ode/sm2-width93-seeds42-44/`. It contains
exact configurations/feature names, pretraining evidence, separate sanity
artifacts, per-seed fresh/selected checkpoints, initial derivative statistics,
histories, gradient norms, dynamics changes, solver counts/sensitivity,
aligned predictions, comparisons/differences, and PNG/SVG learning curves.
Saved split/diagnostic indices and an unchanged preprocessor copy remain private.
`run.json` records original-input/consumed-artifact hashes, reference provenance,
source hashes, and versions. Raw hashes are inherited without reopening data.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/workspace/.cache/matplotlib XDG_CACHE_HOME=/workspace/.cache \
python -m kan_phenotyping.experiments.forecasting_mlp_ode \
  --dataset-dir artifacts/conditional_forecasting/sm2-seed42 \
  --kan-dir artifacts/conditional_forecasting_kan_ode/sm2-fixed-seeds42-44 \
  --matched-dir artifacts/conditional_forecasting_matched_recurrent/sm2-clip5-seeds42-44 \
  --ridge-dir artifacts/conditional_forecasting_feature_ablations/sm2-seed42 \
  --output artifacts/conditional_forecasting_mlp_ode/sm2-width93-seeds42-44
python -m pytest -q
```

Use the configured environment and a fresh output directory.

**All 481 repository tests passed**, including **21 new MLP-control tests**
and the **34 unchanged RK4 tests**. New checks cover exact standard/scaled
initialization, parameter counts, the unchanged inherited forward schedule,
zero-dynamics recurrence recovery, shared initialization guards, all-component
gradients, prefix isolation/state independence, derivative measurement without
weight/RNG changes, deterministic tiny fits/restoration, unbounded output,
metric difference direction, and preserved solver-failure evidence.

**Forty-six fresh-process actual-run audit checks passed.** The audit imports
the model and original data utilities without importing the new experiment
runner. It independently reconstructs fresh initialization, reproduces initial
derivative statistics, restores every selected checkpoint, and exactly matches
all **3 × 851** saved predictions and manually computed MAE/RMSE/R². It replays
checkpoint selection/patience, reconciles optimizer and solver counts, recomputes
first-batch pre-clipping norms without optimizer updates, checks step-halving
and prefix isolation, and verifies aligned references and difference signs.

All **198 previously tracked files**, **three preceding uncommitted matched-
control files**, and **218 prior artifacts** remain unchanged. Four reserved
payloads and sixteen prior non-initial checkpoints were checked using metadata
only; their contents were not opened or hashed. Git HEAD is unchanged.
Python open-event guards record zero raw-data, calibration/test-payload, or
prior trained-checkpoint open attempts in preflight, training and independent
verification. Separate deliberate guard self-tests are excluded from those
counts. The guard is not an operating-system native-I/O sandbox.

Evidence is saved in the run's `validation_checks.json` and `access_checks.json`;
the audit, preservation snapshot/checker, and guard are retained outside source
control at `/workspace/scratch/forecasting_mlp_ode_validation/`.

## Interpretation and limits

MLP dynamics has lower same-seed RMSE than KAN for seeds 42 and 43, and higher
RMSE for seed 44. The mean values nearly coincide numerically, with different
initialization spreads. No equivalence claim follows from these three seeds.
The MLP mean RMSE is about 0.0056 points above the matched recurrent control,
0.0466 above Ridge B, and 0.0245 above Ridge D. It has slightly lower mean MAE
than the two neural controls. These are descriptive predictive comparisons,
not proof of clinical benefit or recovered clinical dynamics.

The comparison approximately matches dynamics parameter counts and matches
shared initialization, optimization, data, and solver protocol. It leaves
function classes, initialization in function space, and optimization paths
different. It does not establish a general KAN-specific advantage or lack of
advantage. Learned coordinates/continuous paths are not measured clinical
states, and endpoint supervision does not identify their physical meaning.

Only two wearable packets are observed. Repeated quarters within participants
are dependent; pooled metrics weight participants with more quarters more
heavily. Complete-history selection, original release coverage, repeated
validation selection/comparisons, conditional feature availability, and unknown
target recall overlap limit inference. Test generalization and prospective
clinical validity remain unverified.

Work stops after this control experiment. No conformal prediction, calibration/
test evaluation, architecture search, commit, or push is performed.
