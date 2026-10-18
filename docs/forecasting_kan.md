# Tabular KAN for conditional relative-month PHQ-9 forecasting

Run on **2026-10-18**, following the saved
[forecasting dataset](forecasting_dataset.md) and
[baseline protocol](forecasting_baselines.md).

**Finding:** this fixed KAN configuration did **not** outperform Ridge on
validation. Its mean RMSE was **3.9120 ± 0.0038**, compared with Ridge's
**3.8136**. KAN had lower RMSE than persistence, but higher MAE. All three
initializations are reported; no additional tuning was performed.

In simple terms, the model receives enrollment information, the measured
PHQ-9 score at the start of the quarter, and the two historical wearable
summaries. It outputs the measured severity expected at SM3, rather than a
change score or a diagnosis. The cutoff is conditionally SM2; the horizon is
**one relative study-month interval**, not a fixed 30 days.

The research sequence remains **KAN → validated future forecasting → KAN-ODE
→ conformal prediction**. This step trains only the flattened-history
tabular KAN. Forecasting timing remains conditional.

## Frozen data and model

The experiment reuses the exact existing cohort, participant assignments,
fitted preprocessing, input order, and measured SM3 target from
`artifacts/conditional_forecasting/sm2-seed42/`. It does not reconstruct the
CSV, reselect eligible samples, refit preprocessing, or change targets.

| Consumed partition | Samples | Participant prefixes under the release mapping |
|---|---:|---:|
| Training | 5,111 | 2,025 |
| Validation | 851 | 338 |

These are the same validation examples used by persistence, training mean,
and Ridge. All quarters from a participant retain their saved assignment.
Calibration/test inputs and outcomes are **never opened or hashed**. Shared
assignment metadata is read only to verify partition integrity.

The **159 inputs** are exactly Ridge's transformed columns: 48 SM1 wearable
columns, a separate 48-column SM2 block, 62 enrollment demographic columns,
and one standardized measured SM0 score. Month-prefixed feature names and the
flattened concatenation are checked by the existing saved-data loader.
Arrays return to their original float32 representation for PyTorch; there is
no new scaling or transformation. Targets remain measured SM3 PHQ-9 scores
in original units, with shape `[N]`; inputs have shape `[N, 159]`.

The existing `TabularDepressionKAN` and `KANLinear` are reused unchanged:

```text
159 transformed inputs → KANLinear(159, 64) → KANLinear(64, 1) → raw scalar
```

There are **122,945 trainable parameters**: 122,176 in the first layer and
769 in the head. Each edge has a SiLU base term plus 11 fixed-grid cubic
B-spline coefficients. Existing settings are fixed at eight grid intervals,
spline degree three, and grid range `[-3, 3]`; there is no adaptive grid or
input clipping. The scalar regression output has **no sigmoid, clipping,
rounding, or category conversion**. The earlier module documentation mentions
PHQ change, but the mathematical model is target-agnostic; this runner passes
measured SM3 severity explicitly and records that definition in checkpoints.

This is a **flattened-history forecasting KAN**, not a temporal convolution
model, KAN-ODE, or exact replication of Interpretable-MTLNet.

## Training-only overfit sanity check

Before any experimental model was fitted, a separate model received 32
deterministically selected **training** samples. No validation payload was
accepted by the diagnostic. Its seed was 2026; it used full-batch AdamW,
learning rate 0.001, weight decay 0.0001, and MSE.

The pass rule was fixed before running: within 500 optimizer steps, achieve
training RMSE ≤0.25 and MSE ≤1% of initial MSE, with finite/nonzero gradients
and changed parameters.

| Diagnostic | Result |
|---|---:|
| Initial training MSE | 59.222923 |
| Final training MSE | 0.061470 |
| Final training RMSE | 0.247932 |
| Optimizer steps | 112 |
| Elapsed diagnostic optimization time | 0.720 seconds |

**Passed.** Both layers' base weights, spline weights, and biases received
finite, nonzero gradients. The diagnostic's checkpoint, curve, indices, and
metrics are stored in the separate `sanity/` directory. Its model/optimizer
were discarded; each real experiment started from a fresh initialization.
The small-sample result checks optimization, not generalization.

## Fixed experiment settings

| Setting | Value for every experiment |
|---|---|
| Initialization seeds | 42, 43, 44 |
| Optimizer | AdamW |
| Learning rate | 0.001, constant; no scheduler |
| Weight decay | 0.0001 |
| Batch size | 128; final partial batch retained |
| Loss | MSE on raw measured SM3 severity |
| Maximum epochs | 50 |
| Early stopping | 10 consecutive epochs without lower validation RMSE; minimum improvement 0 |
| Checkpoint selection | Lowest validation RMSE within each seed; restore before evaluation |
| Minibatch shuffle seed | 42 for every run, separate from initialization |
| Gradient clipping | None; hooks reject any nonfinite gradient |
| Device and precision | CPU, two PyTorch threads, float32, deterministic algorithms |

The existing `Trainer` supplies AdamW, MSE, learning histories, early stopping,
and best-checkpoint restoration. A separate runner supplies the saved tabular
data and gradient checks. Keeping batch order fixed isolates initialization
variation on this split. No architecture, optimizer, or seed search was run.

## Validation results

MAE and RMSE are in **PHQ-9 points**. R² is dimensionless. All metrics use the
same 851 samples, with equal weight per quarter observation.

| Model | Validation MAE | Validation RMSE | Validation R² |
|---|---:|---:|---:|
| Persistence | 2.8496 | 4.0352 | 0.5743 |
| Training mean | 5.0162 | 6.1909 | −0.0020 |
| Ridge, alpha 10 | **2.8209** | **3.8136** | **0.6198** |
| KAN, initialization 42 | 2.9626 | 3.9160 | 0.5991 |
| KAN, initialization 43 | 2.9613 | 3.9085 | 0.6006 |
| KAN, initialization 44 | 2.9489 | 3.9116 | 0.6000 |
| KAN mean ± sample SD | **2.9576 ± 0.0075** | **3.9120 ± 0.0038** | **0.5999 ± 0.0008** |

The SD uses **`ddof=1` across the three initializations**. It describes variation
from initial weights on this one fixed dataset/split. It is **not** a
population confidence interval, patient-level uncertainty, or evidence about
new cohorts. No seed was selected as an overall winner or ensembled.

| Initialization | Selected epoch | Epochs run | Training wall time |
|---:|---:|---:|---:|
| 42 | 13 | 23 | 10.603 seconds |
| 43 | 13 | 23 | 6.864 seconds |
| 44 | 13 | 23 | 7.619 seconds |
| Total experiment fitting | — | 69 | **25.087 seconds** |

Times cover each `Trainer.fit` call, including training, per-epoch validation,
checkpoint/history writing, and its final restoration. They exclude dataset
loading, sanity checking, independent reload verification, and plotting.
These local CPU wall times are not a hardware benchmark.

Every run's best epoch was 13. Training loss continued falling afterward,
while validation did not improve for ten epochs, so all runs stopped at 23.
Learning curves are saved as both PNG and SVG. The plotted training curve is
the square root of online epoch-average training MSE, measured as weights
change; the validation curve evaluates the frozen end-of-epoch model. It is
not a separate post-epoch training evaluation.

Ridge is better than every KAN seed on all three validation metrics. Compared
with persistence, KAN's RMSE is lower but its MAE is higher, so it does not
uniformly improve on “no change.” This step does not isolate wearable value:
any difference may involve SM0 adjustment, demographics, wearables, or their
interactions. No wearable-specific benefit is claimed.

## Saved artifacts and reproduction

New runner: `src/kan_phenotyping/experiments/forecasting_kan.py`.
All existing models, trainers, retrospective workflows, baseline results,
dataset artifacts, and earlier documents are unchanged.

The ignored output directory is
`artifacts/conditional_forecasting_kan/sm2-fixed-seeds42-44/`:

- `config.json`: model, all fixed per-seed settings, conditional assumptions,
  eligibility, original split metadata, and complete input feature names.
- `sanity/`: training-only diagnostic checkpoint, subset sample indices,
  learning history, optimizer settings, elapsed time, and pass evidence.
- `seed_42/`, `seed_43/`, `seed_44/`: `best.pt`, `training_config.json`,
  `history.json`, `validation_metrics.json`, and aligned
  `validation_predictions.npz` with sample indices and measured targets.
- `summary.json`: all seed results, mean/sample SD, unchanged baseline
  comparison, parameter count, timing, and diagnostic results.
- `preprocessor.joblib`: exact copy of the original training-fitted artifact.
- `used_split_indices.npz`: exact original training/validation indices.
- `learning_curves.png` and `learning_curves.svg`: all three learning curves.
- `run.json`: dataset manifest hash, inherited raw input hashes, consumed
  training/validation artifact hashes, source/checkpoint/output hashes,
  versions, deterministic settings, and thread environment.
- `access_checks.json` and `validation_checks.json`: guarded execution and
  independent aggregate verification reports.

From the repository, using the configured Python environment and a **fresh**
output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m kan_phenotyping.experiments.forecasting_kan \
  --dataset-dir artifacts/conditional_forecasting/sm2-seed42 \
  --baseline-dir artifacts/conditional_forecasting_baselines/sm2-seed42 \
  --output artifacts/conditional_forecasting_kan/sm2-fixed-seeds42-44
```

No CLI flags enable a tuning search or reserved-partition evaluation. The
runner refuses to overwrite existing outputs. Raw CSV hashes are inherited
from the validated dataset manifest; the full CSV is not reopened.

## Checks and remaining limits

**208 selected tests passed**, including 23 new forecasting-KAN tests and
185 existing forecasting, data, model, trainer, and metric checks. Tests cover
the exact architecture and AdamW settings, raw outputs, input/target shapes,
finite gradients and rejection of infinite gradients, fixed batch order,
same-seed reproducibility, training-only weights despite changed validation
targets, preprocessing preservation, checkpoint/prediction restoration, and
sample SD. No other real-data model configuration was fitted by these tests.

**Eleven independent uploaded-data checks passed.** In a fresh process, each
checkpoint reproduces saved validation predictions and metrics exactly;
each minimum-validation epoch and patience interval is verified against its
learning history. All six parameter tensors received finite/nonzero gradients
in every run. Input, source, preprocessor, checkpoint, and curve hashes match.
The recorded mean/SD and baseline/sample alignment were independently checked.

File-open guards during training and independent restoration blocked any
calibration/test input or outcome access. Only the original training/validation
observations and shared manifest/mapping/preprocessing metadata were opened.
Hash/metadata preservation checks cover all **57 pre-existing source/document
files and 127 pre-existing artifact files**. Verification scripts are retained
outside Git at `/workspace/scratch/forecasting_kan_validation/`.

The original four conditional assumptions remain: authentic release/export
index correspondence, historical feature availability by SM2 without future
backfill, earlier measured SM0 availability, and enrollment demographics
without later replacement. Actual availability times, survey/device delays,
per-feature construction provenance, and separation from the SM3 PHQ-9
two-week recall period remain unverified. This is not prospective clinical
validation or a certified wholly future symptom interval.

Repeated quarters within participants are dependent. Group separation prevents
cross-partition participant overlap, but pooled metrics weight participants
with more eligible quarters more heavily. Complete-history inclusion and the
released cohort's future-endpoint coverage filter can introduce selection
bias. Validation checkpoint selection is optimistic; untouched test
generalization remains unknown. Fixed grid/range choices and a parameter count
large relative to the training sample count also limit interpretation.

No ODE or conformal method was implemented, no calibration/test performance
was inspected, and no commit or push was made. This step stops after the
prescribed training and validation.
