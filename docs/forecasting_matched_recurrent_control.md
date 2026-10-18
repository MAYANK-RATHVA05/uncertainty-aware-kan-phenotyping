# Optimization-matched non-ODE recurrent control

Experiment date: **2026-10-18**. This step retrains the existing recurrent
control with the completed KAN-ODE experiment's optimizer settings, including
gradient norm clipping at 5. The original
[unclipped recurrent results](forecasting_sequential_kan.md) and
[latent KAN-ODE results](forecasting_kan_ode_results.md) remain unchanged.
The research direction remains **KAN → validated future forecasting → KAN-ODE
→ conformal prediction**.

**Finding:** the clipping-matched recurrent control has validation RMSE
**3.8326 ± 0.0090**, versus the unchanged KAN-ODE's **3.8381 ± 0.0200**.
KAN-ODE minus control RMSE is **+0.0056 points on average**, with mixed signs
across seeds. There is no consistent gain across seeds and metrics. These
are descriptive comparisons; population uncertainty and clinical relevance
are not established. Initial component matching was established for every seed.

In simple terms, this is the same model that reads the two wearable summaries
one at a time and updates a 16-value memory. It has no continuous dynamics.
Changing clipping makes its optimization settings match the ODE experiment;
the ODE model still contains an extra dynamics network and more parameters.

## Frozen data, target, and unchanged model

The exact saved cohort, participant assignments, feature order, measured SM3
target, and training-fitted preprocessing are reused from
`artifacts/conditional_forecasting/sm2-seed42/`. No cohort is rebuilt,
participant reassigned, preprocessing fitted/transformed again, or raw data
reopened. The loader verifies source/input hashes, participant separation,
same-quarter SM1/SM2 positions, and exact reconstruction of the saved inputs.

| Partition | Samples | Participant prefixes | Wearable history | Static context | Target |
|---|---:|---:|---|---|---|
| Training | **5,111** | **2,025** | `[5111,2,48]` | `[5111,63]` | `[5111]` |
| Validation | **851** | **338** | `[851,2,48]` | `[851,63]` | `[851]` |

SM1 precedes SM2. Each packet has the same 27 wearable Statistics values and
21 missingness indicators. The static block contains 62 transformed enrollment
demographic columns and the standardized measured SM0 score paired to that
quarter. The target is measured **SM3 PHQ-9 severity**, in original units,
not change or a generated label. SM3 wearable/intervention inputs and
undocumented regression summaries remain excluded.

The **unchanged 23,169-parameter model** uses a static KAN `63→16`, one shared
packet KAN `48→16`, one shared GRUCell of width 16, and a KAN head `16→1`.
Context initializes the state once; SM1 and SM2 update it in order. The existing
fixed-grid cubic B-splines plus SiLU and ordinary GRU gates are preserved.
Output is unbounded, without prediction clipping, rounding, or a sigmoid.
There is no ODE solver or dynamics module in the control.

Timing stays conditional: release/export index correspondence, historical
summaries available by SM2 without future backfill, earlier measured SM0
availability, and enrollment demographics without later replacement remain
assumptions. SM2→SM3 is **one relative study-month interval**, not 30 days.
Exact available-at timestamps, overlap with the questionnaire's two-week recall
period, and prospective clinical validity remain unverified. Calibration/test
input and outcome payloads remain unused.

## Initial component matching and fixed optimization

Before constructing any optimizer, the runner checks **all three seeds**
against hash-verified ODE `seed_<seed>/initial.pt` artifacts. Fresh recurrent
models are built after resetting each seed; no ODE weights are copied into
them. For every shared state entry, dtype, finiteness, and `torch.equal` are
checked. A mismatch stops the experiment before training or interpretation.

| Seed | Static encoder | Packet encoder | GRU | Output head | Shared state tensors |
|---:|---|---|---|---|---:|
| 42 | Exact | Exact | Exact | Exact | 16 / 16 |
| 43 | Exact | Exact | Exact | Exact | 16 / 16 |
| 44 | Exact | Exact | Exact | Exact | 16 / 16 |

Each seed matches **13 parameter tensors plus three spline-knot buffers**,
including all weights and biases. The only extra ODE state keys are its
dynamics layer's base/spline weights, bias, and knot buffer. All shared tensors
are finite float32. Matching is rechecked on each actual model before fitting,
and fresh recurrent initial checkpoints are retained as evidence.

This matching is possible because the ODE constructor initializes the inherited
static encoder, packet encoder, GRU, and head **before** appending dynamics.
Extra random draws for dynamics cannot change shared components. The minibatch
generator is separately seeded, so those draws also do not change data order.
Only **initial ODE artifacts** are read; no trained ODE checkpoint is opened,
restored, resumed, or used to initialize the control.

| Setting | Matched recurrent | Existing KAN-ODE |
|---|---|---|
| Initialization seeds | 42, 43, 44; fresh shared components verified exactly | Same seeds |
| Optimizer | AdamW | Same |
| Learning rate / weight decay | 0.001 / 0.0001 | Same |
| Batch size / loss | 128 / MSE on measured SM3 | Same |
| Maximum epochs / patience | 50 / 10 | Same |
| Checkpoint selection / minimum improvement | Minimum validation RMSE / 0 | Same |
| Gradient norm clipping | **5** | **5** |
| Minibatch-order seed | **42**, independently of initialization seed | Same |
| Device / precision / threads | CPU / float32 / 2 | Same |
| Deterministic algorithms / workers / final partial batch | Enabled / 0 / retained | Same |
| Learning-rate schedule / architecture search | None / none | Same |
| Parameters | **23,169** | **26,257** |

Every actual ODE `training_config.json` is compared for exact equality with
the control configuration. Runtime versions and original data/source hashes
must also match. Both workflows use the unchanged `make_sequential_loader`
with fixed shuffle seed 42; overlapping epochs have identical sample order.
There are 40 optimizer updates and seven validation batches per epoch.
The existing trainer performs AdamW and clipping unchanged and restores the
minimum-RMSE checkpoint before final validation. No KAN-ODE training or tuning,
additional overfit run, or architecture search occurs.

## Results and aligned comparisons

| Matched-control seed | Validation MAE | Validation RMSE | Validation R² | Best epoch | Stopped epoch | Fit wall time |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2.882038 | 3.837080 | 0.615102 | 23 | 33 | 11.127 s |
| 43 | 2.855635 | 3.838400 | 0.614837 | 20 | 30 | 9.971 s |
| 44 | 2.855940 | 3.822204 | 0.618080 | 20 | 30 | 10.117 s |
| **Mean ± sample SD** | **2.864537 ± 0.015157** | **3.832561 ± 0.008994** | **0.616006 ± 0.001801** | — | 93 total epochs | **31.215 s total** |

All three runs stopped after ten epochs without validation improvement. Fit
wall time includes training, epoch validation, gradient evidence writing,
checkpoint/history writing, and restoration. It excludes initial-artifact
verification, independent checks, final prediction reproduction, and plotting.
These local CPU times are not hardware benchmarks; the models also stop after
different numbers of epochs.

| Aligned model | Validation MAE | Validation RMSE | Validation R² |
|---|---:|---:|---:|
| Historical recurrent, **unclipped** | 2.8517 ± 0.0269 | 3.8264 ± 0.0125 | 0.6172 ± 0.0025 |
| **Matched recurrent, clip 5** | **2.8645 ± 0.0152** | **3.8326 ± 0.0090** | **0.6160 ± 0.0018** |
| Existing KAN-ODE, clip 5 | 2.8615 ± 0.0207 | 3.8381 ± 0.0200 | 0.6149 ± 0.0040 |

| Seed | KAN-ODE MAE / RMSE / R² | Δ MAE: ODE − matched | Δ RMSE: ODE − matched | Δ R²: ODE − matched |
|---:|---|---:|---:|---:|
| 42 | 2.838542 / 3.853956 / 0.611709 | −0.043496 | +0.016876 | −0.003393 |
| 43 | 2.878655 / 3.844812 / 0.613549 | +0.023020 | +0.006412 | −0.001288 |
| 44 | 2.867292 / 3.815639 / 0.619391 | +0.011353 | −0.006565 | +0.001311 |
| **Mean difference** | — | **−0.003041** | **+0.005575** | **−0.001123** |

All metrics use exactly the same 851 validation examples, with equal weight
per quarter sample. MAE/RMSE are in PHQ-9 points; R² is dimensionless. Mean ±
sample SD uses `ddof=1` over three initialization seeds on one fixed split.
Three seeds are **not independent participant samples**, population uncertainty,
confidence intervals, or evidence of statistical significance.

Per-seed differences are explicitly **KAN-ODE metric minus matched-control
metric**. Positive MAE/RMSE differences favor the recurrent control; positive
R² differences favor KAN-ODE. No seed is selected as an overall winner and no
ensemble or follow-up tuning is performed.

## Numerical, clipping, and checkpoint checks

Gradient hooks reuse `GradientDiagnostics`: global/component norms are captured
before clipping, using float64 for observation only. They do not change the
float32 gradients, loss, optimizer, or update order. Optimizer hooks complete
the captured records and reject nonfinite parameters. Predictions, losses,
gradients, and initial state values are checked for finiteness.

All **13 trainable parameter tensors** received finite, nonzero gradients in
each run. Initial weights, training/validation predictions, losses, and updated
parameters remained finite. There were **zero numerical failures**.

| Seed | Optimizer updates | Updates above norm 5 | Clipping frequency | Pre-clipping norm min / mean / max |
|---:|---:|---:|---:|---|
| 42 | 1,320 | 1,320 | 100% | 5.723 / 35.463 / 138.632 |
| 43 | 1,200 | 1,200 | 100% | 5.125 / 36.139 / 128.530 |
| 44 | 1,200 | 1,200 | 100% | 5.756 / 31.950 / 113.876 |
| **Total** | **3,720** | **3,720** | **100%** | — |

Thus clipping was active at every update, as it was in the existing ODE runs.
The global threshold and optimizer settings match; the architectures produce
different gradients and optimization trajectories, which are not forced to
match. Per-update norms and epoch summaries are retained separately.

Every selected checkpoint is restored into a fresh recurrent model and must
reproduce saved predictions and metrics exactly. The shared packet/GRU order,
static initialization once, and SM1's computational isolation from SM2 are
checked. The architecture carries no hidden state across samples, batches, or
forward calls. These computation checks do not validate actual availability
timestamps or clinically meaningful trajectories.

Existing ODE performance is recomputed from its aligned saved **validation
predictions**, with sample indices and measured targets checked exactly. This
does not load trained ODE weights or rerun the ODE model. Initial-artifact,
configuration, prediction, metric, source and data hashes are recorded. Any
initialization, numerical, alignment, or restoration failure stops execution
and preserves evidence rather than silently changing settings.

## Artifacts, reproduction, and preservation

The new reusable runner is
`src/kan_phenotyping/experiments/forecasting_matched_recurrent.py`. Existing
model, trainer, gradient diagnostics, saved loaders, strict guards, and prior
results remain unchanged. The separate ignored output directory is
`artifacts/conditional_forecasting_matched_recurrent/sm2-clip5-seeds42-44/`.

It contains exact configurations/feature names, the all-seed initialization
preflight, checked ODE reference metrics/provenance, saved training/validation
indices, an unchanged preprocessor copy, summary and per-seed differences,
learning curves, and source/input/artifact hashes. Each seed has initial and
selected checkpoints, initialization-match evidence, epoch histories,
per-update pre-clipping norms, clipping frequencies, runtime, numerical checks,
and aligned validation predictions. Original raw hashes are inherited from
the dataset manifest, without reopening raw data.

Using the configured Python environment and a **fresh** output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/workspace/.cache/matplotlib XDG_CACHE_HOME=/workspace/.cache \
python -m kan_phenotyping.experiments.forecasting_matched_recurrent \
  --dataset-dir artifacts/conditional_forecasting/sm2-seed42 \
  --ode-dir artifacts/conditional_forecasting_kan_ode/sm2-fixed-seeds42-44 \
  --output artifacts/conditional_forecasting_matched_recurrent/sm2-clip5-seeds42-44
python -m pytest -q
```

**All 460 repository tests passed**, including **38 new synthetic matched-control
tests**. They check initialization/path/dtype mismatches, fixed settings,
minibatch identity, frozen preprocessing, deterministic tiny fits, independently
observed pre-clipping norms, clipping counts, restoration/alignment, metric
difference direction, provenance guards, and preserved numerical-failure
evidence. Synthetic ODE construction verifies initialization only; it does not
train the real-data ODE experiment.

**Thirty-one independent actual-run checks passed**, using fresh-process direct
recurrent construction rather than the new experiment runner. All **3 × 851**
saved predictions and manually computed MAE/RMSE/R² reproduce exactly. The
audit rechecks all initial components, actual configurations, early stopping,
clipping counts, and first-minibatch pre-clipping norms without optimizer
updates. It independently reproduces every minibatch for two epochs/all seeds
using a seeded index DataLoader, and verifies all paired difference signs and
values from aligned saved ODE predictions.

There were zero recorded attempts to open raw data, calibration/test payloads,
or trained ODE checkpoints in the guarded experiment and independent audit.
The file-open guard covers Python open events, not arbitrary native I/O.
All **198 prior tracked files** and **176 prior artifacts** remain unchanged;
four reserved payloads and thirteen non-initial checkpoints were checked with
metadata only, without reading or hashing their contents. Git HEAD is unchanged.
Aggregate evidence is saved as `validation_checks.json` and `access_checks.json`
inside the ignored run directory. Independent scripts and original audit
reports are retained at
`/workspace/scratch/forecasting_matched_recurrent_validation/`.

## Interpretation and stopping point

On RMSE, the matched recurrent control is lower for seeds 42 and 43; KAN-ODE
is lower for seed 44. KAN-ODE has a slightly lower **mean MAE** but a slightly
higher **mean RMSE** and lower **mean R²**. This does not establish a consistent
improvement from adding dynamics. No setting is adjusted in response to these
results. The matched control's mean RMSE is 0.0061 points above the historical
unclipped control; that earlier result is retained as context, not substituted
for the primary matched comparison.

Matching optimization removes the clipping mismatch. The ODE still has
**3,088 additional parameters**, different forward computation and gradients,
and continuous latent evolution. This is **not a capacity-matched comparison**.
Differences remain descriptive; they do not identify a causal effect of
dynamics, KAN-specific benefit, clinical trajectory recovery, methodological
novelty, or a clinically meaningful difference.

Only two historical wearable packets are observed. Complete-history and
release-coverage selection, dependent repeated quarters, and heavier weighting
of participants with more quarters limit interpretation. Repeated validation
selection/comparisons can create optimism. Prospective timing and test
generalization remain unknown. The original unclipped results remain a
separate historical experiment; they are not replaced by these runs.

Work stops after this matched-control comparison. No MLP-ODE, conformal
prediction, test evaluation, architecture search, commit, or push is performed.
