# Controlled Ridge feature ablations for conditional forecasting

Run on **2026-10-18**, after reading the
[dataset](forecasting_dataset.md), [baseline](forecasting_baselines.md), and
[KAN](forecasting_kan.md) reports.

**Finding:** enrollment demographics improve validation RMSE beyond measured
SM0 alone. Adding wearable packets does **not** improve RMSE beyond SM0 plus
demographics in this Ridge comparison. SM2 slightly improves MAE while worsening
RMSE; adding SM1 worsens both. These are exploratory predictive comparisons,
not causal effects or evidence of statistical significance.

The research sequence remains **KAN → validated future forecasting → KAN-ODE
→ conformal prediction**. No KAN tuning or later-stage implementation was
performed, and all previous workflows and results are preserved.

## Frozen sample and fitting protocol

The target is **measured SM3 PHQ-9 severity**, not change or a generated label.
Inputs are conditionally available by **SM2**, with a horizon of **one relative
study-month interval**, not a fixed 30 days. All four previously recorded
availability/export assumptions remain in effect.

The runner loads the exact saved forecasting bundle at
`artifacts/conditional_forecasting/sm2-seed42/`. It does not reopen the full CSV,
rebuild the cohort, regenerate participant assignments, or refit/transform
preprocessing. It subsets existing transformed columns only.

| Partition used for every setup | Samples | Participant prefixes under the release mapping |
|---|---:|---:|
| Training | **5,111** | **2,025** |
| Validation | **851** | **338** |

Every setup uses identical sample indices, ordering, and measured targets.
All quarters from each participant stay in their original partition. Even A
and B retain the original complete-history cohort; removing wearable inputs
does **not** admit participants or quarters previously excluded for missing
historical rows. This controls sample composition across comparisons.

Each setup evaluates exactly the original alpha grid
**`[0.01, 0.1, 1, 10, 100]`** and selects minimum validation RMSE. Exact ties
retain the first alpha in ascending order. Ridge uses the existing float64,
SVD-solver configuration with a fitted intercept and no coefficient-sign
constraint. Every candidate is fitted on training data only; there is no
train-plus-validation refit. Predictions remain unbounded and unrounded;
targets remain in their original units.

Calibration/test input and outcome payloads are **never opened or hashed**.
Shared assignment metadata is used only to verify split integrity. Raw input
hashes are inherited from the dataset manifest; consumed train/validation and
preprocessing hashes are verified and recorded.

## Complete original-feature selection

| Setup | Inputs | Original-feature/month groups | Transformed columns |
|---|---|---:|---:|
| A | Measured SM0 PHQ-9 only | 1 | 1 |
| B | SM0 + enrollment demographics | 30 | 63 |
| C | SM0 + demographics + SM2 wearable packet | 57 | 111 |
| D | SM0 + demographics + SM1 and SM2 wearable packets | 84 | 159 |

**A is a fitted SM0-only Ridge**, which learns an intercept and regularized
slope from training outcomes. It is different from persistence's fixed
`SM3 = SM0` rule. Its one predictor is the unchanged training-standardized
representation of the measured SM0 score, with no additional information.

Selection uses the fitted `ColumnTransformer` output positions, original
column lists, imputer `indicator_.features_`, and encoder vocabulary spans.
It does not infer ownership by splitting or substring-matching feature names.
This matters because several original names contain repeated underscores.

The original transformed order is retained:

| Block | Saved flat indices, zero-based | Composition |
|---|---|---|
| SM1 | 0–47 | 27 wearable values + 21 training-derived missingness indicators |
| SM2 | 48–95 | The same 27 fields and 21 indicators, separately represented |
| Enrollment demographics | 96–157 | 24 numeric values, six numeric missingness indicators, 32 one-hot columns for five categorical originals |
| Measured SM0 PHQ-9 | 158 | One standardized measured score, never imputed |

A uses index 158; B uses 96–158; C uses 48–158; D uses 0–158. These positions
are **derived and checked against fitted metadata**, not hardcoded in the
runner. All 159 columns have exactly one original-feature/month owner.
Whenever an original feature enters a setup, all its associated indicators
and fitted one-hot categories enter together. All other original groups are
entirely absent. Thus A ⊂ B ⊂ C ⊂ D, with SM1/SM2 identity preserved.

No SM3 wearable/intervention input, Dynamic change feature, undocumented
regression-summary field, generated PHQ label, or identifier is introduced.

## Validation comparison

MAE and RMSE are in **PHQ-9 points**; R² is dimensionless. Metrics give each
quarter sample equal weight. Alpha is selected by RMSE, even when another
candidate has a slightly lower MAE.

| Setup | Selected alpha | Validation MAE | Validation RMSE | Validation R² |
|---|---:|---:|---:|---:|
| A: SM0 only | 0.01 | 2.8751 | 3.8648 | 0.6095 |
| B: A + demographics | 10 | 2.8087 | **3.7915** | **0.6242** |
| C: B + SM2 | 10 | **2.7977** | 3.8003 | 0.6224 |
| D: C + SM1, original full Ridge | 10 | 2.8209 | 3.8136 | 0.6198 |

The following deltas are **later setup minus earlier setup**. Negative
MAE/RMSE and positive R² favor adding the feature bundle.

| Comparison | Added bundle | ΔMAE | ΔRMSE | ΔR² |
|---|---|---:|---:|---:|
| B − A | Enrollment demographics | −0.06635 | −0.07332 | +0.01468 |
| C − B | Latest wearable packet, SM2 | −0.01103 | +0.00883 | −0.00175 |
| D − C | Additional historical packet, SM1 | +0.02322 | +0.01329 | −0.00265 |

- **B versus A:** demographics improve all three validation metrics for this
  Ridge protocol. The RMSE reduction is about 0.0733 PHQ-9 points.
- **C versus B:** SM2 yields a mixed result: average absolute error decreases
  by about 0.0110 points, while RMSE increases by about 0.0088 points and R²
  decreases. It does not improve the primary selection metric.
- **D versus C:** adding SM1 worsens MAE by about 0.0232 points and RMSE by
  about 0.0133 points; it supplies no incremental validation gain here.

B has the lowest RMSE and C the lowest MAE among these four comparisons.
The changes are small. Each setup independently selects its alpha, so these
are bundle comparisons under the prescribed selection procedure, not
fixed-penalty coefficient effects. No significance test, confidence interval,
or causal claim follows from these differences.

## Reuse and reproduction of D

D is the existing full Ridge from
`artifacts/conditional_forecasting_baselines/sm2-seed42/`.
Its dataset-manifest identity, original input hashes, target, assumptions,
159 feature names, optimizer-free Ridge configuration, and prediction/sample
alignment were checked for compatibility.

All five D candidates were independently fitted on the original training
matrix to verify the saved coefficients, intercepts, predictions, metrics,
and selected alpha. The declared numerical tolerance was absolute `1e-10`;
the **observed maximum prediction difference was exactly zero**, for every
candidate. Alpha 10 was selected again.

The reported D metrics, selected model, candidate models, and predictions
were then **reused from the original artifacts**. Their copies/derivatives are
saved under the new comparison directory. No original baseline artifact was
replaced. Fifteen new candidate fits cover A–C; the five additional D fits
serve reproduction, not a new search or replacement result.

## Saved artifacts and reproduction

New implementation:
`src/kan_phenotyping/experiments/forecasting_feature_ablations.py`.

The ignored output directory is
`artifacts/conditional_forecasting_feature_ablations/sm2-seed42/`:

- `A/`, `B/`, `C/`, `D/`: exact `features.json` lists and original-feature
  group mappings, `selected_ridge.joblib`, all five `ridge_candidates.joblib`,
  `validation_metrics.json`, and `validation_predictions.npz` containing
  sample indices, measured SM3 targets, and predictions.
- `D/reproduction.json`: original artifact references/hashes, compatibility
  checks, alpha verification, and numerical reproduction differences.
- `feature_groups.json`: all 84 original-feature/month owners and their
  exact transformed column positions/names.
- `summary.json` and `config.json`: complete comparison, signed increments,
  candidate grid, fixed fitting protocol, assumptions, counts, and lineage.
- `preprocessor.joblib`: unchanged copy of the saved fitted transform.
- `used_split_indices.npz`: original training/validation sample indices.
- `run.json`: inherited input hashes, consumed artifact/source hashes,
  original D references, output hashes, software versions, and thread settings.
- `access_checks.json` and `validation_checks.json`: aggregate runtime-access
  and independent verification reports.

From the repository, using the configured Python environment and a **fresh**
output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m kan_phenotyping.experiments.forecasting_feature_ablations \
  --dataset-dir artifacts/conditional_forecasting/sm2-seed42 \
  --baseline-dir artifacts/conditional_forecasting_baselines/sm2-seed42 \
  --output artifacts/conditional_forecasting_feature_ablations/sm2-seed42
```

The runner refuses to overwrite existing outputs or write generated arrays
to source directories. All sample-level data and fitted artifacts stay outside
tracked source; only aggregate results are included in this report.

## Checks and interpretation limits

**139 selected tests passed:** 20 new ablation checks and the 119 existing
forecasting dataset/baseline/data/inspection checks. They verify complete
indicator/one-hot groups, nested feature sets, month identity, training-only
fits for every alpha, excluded-SM1 counterfactual invariance, validation targets
not affecting fitted coefficients, identical sample alignment, deterministic
repetition, saved feature/model/prediction consistency, compatible D reuse,
and rejection of partial/mislabeled groups or modified original artifacts.
Guarded tests forbid preprocessing fitting/transformation and reserved-payload
opens. No KAN training tests or real-data KAN runs were invoked in this step.

**Ten independent actual-data checks passed.** An independent repeat reproduced
all metrics, predictions, selected alphas, and candidate coefficients.
Freshly loaded models reproduce the saved aligned predictions, and original
D outputs match exactly. Source/input/output/preprocessor hashes match.
The original selected D model also matches its saved winning candidate.
Preservation checks cover **60 pre-existing source/document files and 155
pre-existing artifact files**, including the dataset, retrospective, baseline,
and KAN results. Runtime guards in both execution and independent verification
record **zero calibration/test payload opens**. Verification scripts are saved
outside Git at `/workspace/scratch/forecasting_feature_ablations_validation/`.

These comparisons apply to one selected complete-history cohort and one
participant-disjoint validation split. Quarters from the same participant
remain dependent, and pooled metrics weight participants with more eligible
quarters more heavily. Reusing validation for alpha selection and repeated
feature/model comparisons creates optimism. The release's future-endpoint
coverage filter and complete-history requirement can also introduce selection
bias. No untouched-test generalization result is available.

Conditional timing remains unresolved: release/export correspondence,
historical feature availability without future backfill, earlier SM0 score
availability, and enrollment measurements without later replacement are
assumptions. Actual survey/device delays, feature provenance, and separation
from the target's two-week recall period are unverified. This is not
prospective clinical validation.

Most importantly, a small or adverse Ridge increment does **not** show that
wearables contain no useful information. Linear additive models, this feature
representation, regularization, missingness, sample selection, and measurement
noise may limit the observed gain. Useful nonlinear or temporal relationships
remain possible. This step neither tests those alternatives nor changes the
research direction.

No KAN was tuned, no ODE/conformal method was implemented, and no calibration
or test performance was inspected. No commit or push was made. Work stops
after this controlled comparison.
