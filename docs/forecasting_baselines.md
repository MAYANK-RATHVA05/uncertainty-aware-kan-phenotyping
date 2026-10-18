# Baselines for conditional relative-month PHQ-9 forecasting

Run on **2026-10-18**, using the saved dataset described in
[forecasting_dataset.md](forecasting_dataset.md). The target is **measured SM3
PHQ-9 severity**, with information conditionally available by **SM2**. The
horizon is **one relative study-month interval**, not a fixed 30 days.

**Finding:** Ridge modestly improves on predicting no change in the validation
partition. Persistence is substantially stronger than the training-mean
baseline. These are validation findings under conditional timing assumptions;
no calibration or test performance was computed.

The research sequence remains **KAN → validated future forecasting → KAN-ODE
→ conformal prediction**. This step fits only the simple baselines below.

## What the three baselines do

| Baseline | Prediction and purpose |
|---|---|
| Persistence | Predict SM3 severity equal to that quarter's measured SM0 PHQ-9. It asks how difficult “no change” is to beat. No parameters are fitted. |
| Training mean | Predict the same value for everyone: the mean measured SM3 severity among training samples, **7.311485** PHQ-9 points. Validation outcomes never enter that mean. |
| Ridge regression | Fit a regularized linear model using enrollment demographics, measured SM0 PHQ-9, and separate SM1/SM2 wearable blocks. Select its regularization strength using validation RMSE. |

No baseline predicts PHQ change. Predictions are not clipped, rounded, or
converted to categories, and targets remain raw measured severities in 0–27.

## Reused cohort, splits, and preprocessing

The runner consumes the existing saved bundle at
`artifacts/conditional_forecasting/sm2-seed42/`. It **does not** rebuild
eligibility or generate a new participant split. The complete-history cohort
still requires SM1, SM2, and both measured endpoints: 8,494 samples from
3,375 participant prefixes under the release mapping. Individual missing
predictors remain governed by the original training-fitted transform.

| Partition used in this step | Quarter samples | Participant prefixes |
|---|---:|---:|
| Training | **5,111** | **2,025** |
| Validation | **851** | **338** |

All three baselines use the **same 851 validation samples in the same order**.
Saved sample indices accompany predictions. Every quarter for a participant
remains in its original partition; participant assignments across all four
partitions are checked using shared mapping metadata only.

The original calibration and test assignments remain reserved. Their input
and target files are **not opened, hashed, transformed, fitted, or evaluated**
by this runner. Original raw input hashes are inherited from the dataset
manifest; the complete CSV is not reopened. Hashes of the manifest, consumed
training/validation files, split/mapping metadata, and saved preprocessor are
verified and recorded.

The saved preprocessing is reused without refitting. Its medians, scales,
missingness indicators, and category vocabulary were fitted on the original
training predictors only: 5,111 samples and 10,222 historical wearable packets.
No validation outcomes enter preprocessing or model fitting.

Ridge uses the original **159 transformed features**, flattened in this order:

1. SM1's 48 wearable columns, including training-derived missingness indicators.
2. SM2's separate 48 wearable columns, in the same within-month order.
3. The 62 transformed enrollment demographic columns.
4. One standardized measured SM0 PHQ-9 column.

Feature names retain `SM1__` and `SM2__` prefixes. The runner verifies that the
saved flat matrix exactly equals this concatenation. SM3 wearable/intervention
inputs, Dynamic features, undocumented regression summaries, and generated
PHQ labels remain excluded by the original builder's allowlist.

For persistence, the saved training scaler is inverted to recover SM0.
Because the builder stored standardized float32 values, the runner requires
the inverse value to be within `1e-4` of a valid measured integer, then requires
that integer to reproduce the saved standardized value **exactly** on a
float32 round trip. This restores the original input score; it is not rounding
or clipping predictions. Persistence outputs that exact measured integer.

## Validation results and alpha selection

MAE and RMSE are in **original PHQ-9 points**; R² is dimensionless. Each quarter
sample has equal weight. Smaller MAE/RMSE and larger R² indicate better fit.

| Baseline | Validation MAE | Validation RMSE | Validation R² |
|---|---:|---:|---:|
| Persistence: measured SM0 | 2.8496 | 4.0352 | 0.5743 |
| Training mean | 5.0162 | 6.1909 | −0.0020 |
| **Ridge, alpha 10** | **2.8209** | **3.8136** | **0.6198** |

Ridge reduces RMSE by approximately **0.222 points (5.5%)** compared with
persistence. Its MAE improvement is approximately **0.029 points (1.0%)**.
This suggests a modest validation gain over a strong no-change baseline;
it does not establish statistical significance or clinical usefulness.

The requested fixed alpha grid was evaluated once:

| Ridge alpha | Validation RMSE |
|---:|---:|
| 0.01 | 3.815792 |
| 0.1 | 3.815730 |
| 1 | 3.815222 |
| **10** | **3.813633** |
| 100 | 3.815076 |

**Selected alpha: 10**, by minimum validation RMSE. Exact ties would retain
the first alpha in the ascending grid. All five models use training data
only, a fitted intercept, float64 matrices, and scikit-learn Ridge's
deterministic `svd` solver. The selected artifact is the already fitted
training candidate; there is **no train-plus-validation refit**. Candidate
RMSEs are close, so this is not evidence that alpha 10 is uniquely robust.

## Saved artifacts and reproduction

Implementation:
`src/kan_phenotyping/experiments/forecasting_baselines.py`.
The separate retrospective workflow and its saved results are unchanged.

The ignored output directory is
`artifacts/conditional_forecasting_baselines/sm2-seed42/`:

- `config.json`: assumptions, eligibility, feature allowlist/names, baseline
  definitions, alpha grid/selection, and preprocessing/prediction policy.
- `validation_metrics.json`: full-precision comparison, all five candidate
  validation metrics, training constant, and sample/participant counts.
- `models.joblib`: persistence, fitted training mean, and selected Ridge.
- `ridge_candidates.joblib`: all five fitted training-only Ridge candidates.
- `preprocessor.joblib`: unchanged copy of the original fitted preprocessor.
- `validation_predictions.npz`: exact sample indices, measured SM0/SM3, and
  the three aligned prediction vectors; retained privately outside Git.
- `used_split_indices.npz`: the original training/validation sample indices.
- `run.json`: original input hashes, consumed file hashes, source/artifact
  hashes, original split configuration, versions, and thread settings.
- `validation_checks.json`: independent aggregate validation of the real run.

From the repository, using the configured Python environment and a fresh
output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m kan_phenotyping.experiments.forecasting_baselines \
  --dataset-dir artifacts/conditional_forecasting/sm2-seed42 \
  --output artifacts/conditional_forecasting_baselines/sm2-seed42
```

The runner refuses to overwrite existing output or write generated data to
source directories. Models saved through this CLI can be loaded in a fresh
Python process; the custom baselines use their importable module paths.

## Checks and limitations

**119 selected tests passed:** 18 new baseline tests plus the 101 existing
conditional-builder/data/inspection checks. Tests cover exact persistence,
training-only means and all five Ridge fits, no refit during selection,
validation-target changes leaving fitted coefficients unchanged, correct
sample/month order, original-unit unclipped predictions, deterministic
repeats, integrity failures, and fresh-process CLI artifact reload. A guarded
test prevents any opening of reserved calibration/test observation files and
prevents preprocessing refits.

The independent uploaded-data check verifies aligned predictions and metrics,
exact persistence, the training mean, selected candidate, artifact/source
hashes, and an independent repeat with identical metrics, predictions, and all
five sets of Ridge coefficients. A runtime file-open guard also blocks reserved
observation files. Existing source/documents, retrospective results, and all
13 saved dataset artifacts are preserved. The verification script is retained
outside Git at
`/workspace/scratch/forecasting_baselines_validation/validate_run.py`.

The four conditional assumptions from the dataset report still apply:
release/export index correspondence; no-look-ahead historical feature
availability by SM2; earlier availability of measured SM0; and enrollment
demographics without later replacement. Actual available-at timestamps,
survey/device delays, and separation from the target's two-week recall
period remain unverified. These results are not prospective clinical
validation or a forecast of a certified wholly future symptom interval.

Quarter samples repeat within participants. The disjoint split prevents
participant overlap across partitions, but pooled metrics give more weight
to participants with more eligible quarters and do not imply independent
observations or participant-level confidence intervals. Complete-history
selection and the release's future-endpoint coverage filtering may bias the
cohort. Validation was used to choose alpha, so its score has selection
optimism; untouched test generalization remains unknown.

The comparison also does not isolate the value of wearables: Ridge can gain
from adjusting SM0 and using demographics. No claim about incremental
wearable benefit follows from this three-baseline comparison alone.

No KAN was trained, no ODE/conformal method was implemented, and no calibration
or test performance was inspected. No commit or push was made.
