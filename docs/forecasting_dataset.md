# Conditional relative-month forecasting dataset

Implemented and checked on **2026-10-18**. This step prepares data only.
The research sequence remains **KAN → validated future forecasting → KAN-ODE
→ conformal prediction**. No model training, ODE implementation, or conformal
analysis was performed.

In simple terms, each example contains the person's enrollment information,
their measured PHQ-9 score at the start of that quarter, and two historical
wearable summaries. Its separate output is their measured PHQ-9 severity at
the end of the quarter. It does **not** predict score change.

## Decision and explicit assumptions

The **complete** [temporal feasibility report](temporal_feasibility.md) supports
this candidate **conditionally**, using the accepted paper's physical pp. 8–10,
the earlier SSRN pp. 5–8, the pinned official code, and the local dictionary.
Those sources establish the nominal sample-month convention; they do not
verify the export's actual available-at timestamps. No essential structural
mapping was overridden or replaced with a numerical guess.

The separate `ConditionalForecastAssumptions` API defaults to blocked and
requires explicit acknowledgement of all four working assumptions:

1. The CSV preserves the official release's participant/month index mapping.
2. The selected SM1/SM2 wearable Statistics summaries use historical data and
   are computable and available by the SM2 cutoff, without future backfilling.
3. The paired start score is the measured SM0 assessment, available by SM2.
4. The selected Static fields are enrollment screener measurements already
   available by SM2, rather than later replacements.

The acknowledgement and an evidence reference are saved with the artifacts.
They are **assumptions**, not verified timing flags. The existing strict
`ForecastConfig`, guarded seven-month helper, retrospective workflow, and
retrospective results remain unchanged. This new API does not set their flags.

The prediction cutoff is **SM2**, after conditional availability of its packet.
The horizon is **one relative study-month interval**, not a fixed 30 days.
This is not prospective clinical validation, and target recall-period
nonoverlap is not established.

## Exact sample and baseline mapping

For a participant prefix `p` and endpoint suffix `m ∈ {3, 6, 9, 12}`:

| Item | Explicit source |
|---|---|
| Participant | Exact index prefix, preserved as a string; joining/splitting only |
| Quarter | Zero-based `(m − 1) // 3`; never joined across quarters |
| Historical SM1 | Exact key `(p, m − 2)` |
| Historical SM2 | Exact key `(p, m − 1)` |
| Measured SM0 PHQ-9 input | `phq9_score_start` on the same quarter's endpoint row `(p, m)` |
| Enrollment demographic input | Only the SM1 copy of the 29 reviewed Static fields; no later-row backfill |
| Measured SM3 severity target | `phq9_score_end` on `(p, m)`, unscaled integer-valued severity in 0–27 |

The start score's interpretation comes from the dictionary's “start of the
3 month period,” the papers' measured SM0/SM3 assessments, and the official
code's start/end-to-SM0/SM3 mapping. Its storage on an endpoint row does not
make all fields on that row safe predictors. The builder selects **only** the
earlier start-score scalar from that row for predictor construction, and
constructs the target separately. No row 0 is required. Each later quarter
uses its own paired SM0 score, never a single enrollment PHQ repeated over
all quarters or a baseline reconstructed from a previous target.

| Endpoint suffix | Quarter | SM1 suffix | SM2 suffix |
|---:|---:|---:|---:|
| 3 | 0 | 1 | 2 |
| 6 | 1 | 4 | 5 |
| 9 | 2 | 7 | 8 |
| 12 | 3 | 10 | 11 |

Exactly two observed wearable rows and both measured endpoints are required.
Missing individual predictor values are retained. Targets and SM0 scores are
never imputed. Relative positions `[1, 2]` and actual study suffixes are saved;
there is no daily interpolation, padding, or borrowing from another quarter
or participant. Malformed/duplicate keys, out-of-range months, invalid scores,
and measured endpoint scores on intermediate rows cause errors rather than
guessed repairs.

## Fixed feature allowlist

The builder checks the supplied dictionary's names, descriptions, categories,
and subcategories against fixed reviewed names. Newly added dictionary
features are not automatically included.

**Wearable Statistics, 27 fields at each of SM1 and SM2:**

```text
steps_awake_mean
steps_mvpa_iqr
steps_lpa_iqr
steps_awake_sum_iqr
steps__active_day_count_
steps__sedentary_day_count_
steps_mvpa_sum_recent
steps_lpa_sum_recent
steps_rolling_6_median_recent
steps_rolling_6_max_recent
sleep_asleep_weekday_mean
sleep_asleep_weekend_mean
sleep_in_bed_weekday_mean
sleep_in_bed_weekend_mean
sleep_ratio_asleep_in_bed_weekday_mean
sleep_ratio_asleep_in_bed_weekend_mean
sleep_in_bed_iqr
sleep_asleep_iqr
sleep_ratio_asleep_in_bed_iqr
sleep_main_start_hour_adj_median
sleep_main_start_hour_adj_iqr
sleep_main_start_hour_adj_range
sleep__hypersomnia_count_
sleep__hyposomnia_count_
sleep_asleep_mean_recent
sleep_in_bed_mean_recent
sleep_ratio_asleep_in_bed_mean_recent
```

**Enrollment demographics, 29 Static fields:**

```text
sex, race_white, race_black, race_hispanic, race_asian, race_other,
birthyear, educ, height, weight, bmi, pregnant, birth, trauma,
insurance, money, money_assistance, household, comorbid_cancer,
comorbid_diabetes_typ1, comorbid_diabetes_typ2, comorbid_gout,
comorbid_migraines, comorbid_ms, comorbid_osteoporosis,
comorbid_neuropathic, comorbid_arthritis, num_migraine_days, meds_migraine
```

**One additional input:** the measured SM0 `phq9_score_start` scalar.
Demographics remain enrollment measurements even when sample SM0 is later.

Excluded: all SM3 wearable/intervention inputs; all Dynamic intervention/change
features at every month; all 84 undocumented regression-summary fields;
PHQ categories, generated intermediate labels/probabilities, identifiers, and
all other features outside the allowlist. Neither generated PHQ labels nor
their predictions are used as ground truth. The 27 Statistics descriptions
support their selection; their precise operational availability remains an
explicit condition, not something inferred from column order.

## Actual eligibility and exclusions

Counts below are structural eligibility under the assumptions, not certified
real-time availability counts. No individual records are printed.

| Stage | Samples/rows | Distinct participant prefixes |
|---|---:|---:|
| Input CSV | 35,694 rows | 4,948 |
| Existing quarterly endpoint pool | 10,866 samples | 4,036 |
| Eligible: both histories and measured endpoints | **8,494 samples** | **3,375** |

All 10,866 existing endpoint rows have both measured scores: zero exclusions
for missing SM0/SM3 endpoints. The builder would exclude missing pairs in
another export. It does not invent endpoint opportunities for absent quarters.
There are 912 input prefixes without an endpoint in the released pool.

| History present in measured endpoint pool | Samples | Participant prefixes (overlap across rows) |
|---|---:|---:|
| Both SM1 and SM2 | 8,494 | 3,375 |
| SM1 only | 786 | 725 |
| SM2 only | 899 | 815 |
| Neither | 687 | 639 |

Requiring both histories excludes **2,372 endpoint samples (21.8%)** and leaves
**661 endpoint-pool participants** without an eligible sample. History-pattern
participant counts overlap across quarters and must not be summed.

| Endpoint suffix | Measured endpoint pool | Eligible | Missing required history |
|---:|---:|---:|---:|
| 3 | 3,244 | 2,488 | 756 |
| 6 | 2,832 | 2,166 | 666 |
| 9 | 2,732 | 2,157 | 575 |
| 12 | 2,058 | 1,683 | 375 |

Eligible raw predictors retain **56,005 missing wearable cells** across the
two histories and **202 missing demographic cells**. These do not trigger
complete-case exclusion.

## Participant partitions and training-only preprocessing

Default fractions are **60% train / 10% validation / 10% calibration / 20%
test**, allocated by participant with seed **42**. Sorted participant strings
are shuffled with NumPy's seeded generator; largest-remainder rounding breaks
ties in the listed order. All quarters for a participant stay together.
Splitting uses no targets or target stratification. This separate protocol
does not change the retrospective split.

| Partition | Samples | Participant prefixes |
|---|---:|---:|
| Train | 5,111 | 2,025 |
| Validation | 851 | 338 |
| Calibration, reserved | 827 | 337 |
| Test | 1,705 | 675 |
| Total | 8,494 | 3,375 |

All samples appear exactly once; participant intersections are empty across
all six partition pairs. Calibration is saved for later conformal research;
no residuals, quantiles, intervals, or conformal procedure are computed.

`ConditionalForecastPreprocessor.fit` receives only the predictor payload and
training indices. Numeric wearable/demographic values use training medians,
training-discovered missingness indicators, and training scaling. All-missing
training columns remain present with zero imputation. Infinite numeric
predictors are treated as missing. The five demographic categories are `sex`,
`educ`, `money`, `household`, and `num_migraine_days`; their vocabulary is learned
only from training data, missingness has an explicit category, and unseen
held-out categories are ignored. SM0 PHQ uses its own training scaler and
is never imputed. SM3 targets remain raw and never enter preprocessing.

The wearable transform is shared across the **10,222 training packets**.
Only transformation is applied to validation/calibration/test inputs. No
held-out median, scale, missingness indicator, or category is learned.

## Tensor layout and saved artifacts

For `N` samples, raw shapes are wearable `[N, 2, 27]`, demographic `[N, 29]`,
SM0 PHQ `[N, 1]`, relative/study month positions `[N, 2]`, and separate target
`[N]`. Internally, raw wearable DataFrame rows are stacked sample-major,
SM1 then SM2, as `[2N, 27]`.

For this export and split, training-fitted transformations produce:

| Partition | Wearable | Demographics | SM0 PHQ | Optional flattened inputs | Separate SM3 target |
|---|---|---|---|---|---|
| Train | `[5111, 2, 48]` | `[5111, 62]` | `[5111, 1]` | `[5111, 159]` | `[5111]` |
| Validation | `[851, 2, 48]` | `[851, 62]` | `[851, 1]` | `[851, 159]` | `[851]` |
| Calibration | `[827, 2, 48]` | `[827, 62]` | `[827, 1]` | `[827, 159]` | `[827]` |
| Test | `[1705, 2, 48]` | `[1705, 62]` | `[1705, 1]` | `[1705, 159]` | `[1705]` |

Input arrays are float32; month positions remain integer arrays `[N, 2]`.
The 48 wearable columns include 21 training-discovered missingness indicators;
the 62 demographic columns include numeric indicators and one-hot categories.
The optional flat layout retains **both** SM1 and SM2: `2 × 48 + 62 + 1 = 159`.
Exact transformed feature names/order are saved in the manifest.

Generated files are outside tracked source, in the ignored directory
`artifacts/conditional_forecasting/sm2-seed42/`:

- `<partition>_inputs.npz`: only wearable, demographics, SM0 PHQ, relative/study
  positions, and optional flattened inputs; no targets or identities.
- `<partition>_targets.npz`: only measured `sm3_phq`, separately stored.
- `splits.npz`: canonical sample indices for all four partitions.
- `sample_index.npz`: private participant/quarter and source-row mappings;
  identifiers are not predictors or part of this public report.
- `preprocessor.joblib`: serialized training-fitted transforms.
- `manifest.json`: assumptions, exclusions, input/source/artifact hashes,
  versions, seed/fractions, allowlist, feature order, counts, and shapes.
- `validation_checks.json`: aggregate independent export-validation results.

The exporter refuses nonempty output directories and source-directory output.
Input hashes are retained in the ignored manifest rather than hardcoded in
source. Generated identifiers, split definitions, hashes, and serialized
transforms have not been added to Git.

## Reproduction and validation

From the repository, using the configured Python environment and a **fresh**
output directory:

```bash
python -m kan_phenotyping.experiments.build_forecasting_dataset \
  --data /workspace/private-data/psyche-d/anon_processed_df_parquet.csv \
  --dictionary /workspace/private-data/psyche-d/feature_explanations.tsv \
  --output artifacts/conditional_forecasting/sm2-seed42 \
  --seed 42 \
  --acknowledge-conditional-assumptions \
  --assumption-reference docs/temporal_feasibility.md
```

Reusable APIs live in `data/conditional_forecasting.py`,
`data/forecasting_splits.py`, `data/forecasting_preprocessing.py`, and
`experiments/build_forecasting_dataset.py` under `src/kan_phenotyping/`.
`ForecastPredictors` contains no targets; the sample constructor does not pass
the whole endpoint row or target field to its predictor assembler.

Validation includes counterfactual changes to future SM3 wearable/intervention
values, target-only SM3 PHQ changes, quarter-specific SM0 recovery with unequal
consecutive endpoint scores, exact participant/quarter joins after shuffling,
missing histories/endpoints, dictionary/index guards, participant exclusivity,
and held-out changes that cannot alter fitted medians/scales/categories.
The existing strict forecasting guard is checked to remain blocked by default.
Artifact tests check separate target storage, reload reconstruction, hashes,
reproducibility, and output safeguards.

The uploaded-data check independently verifies all sample mappings, all four
persisted partitions, targets against measured endpoint columns, and every
saved input against the reloaded transform. A second independent export checks
identical arrays, split definitions, metadata, and fitted statistics; each
export's file hashes are independently verified. The validation script is retained
outside Git at
`/workspace/scratch/forecasting_builder_validation/validate_artifacts.py`.

Test command (data tests only; model-training tests are not invoked):

```bash
python -m pytest -q \
  tests/test_conditional_builder_validation.py \
  tests/test_conditional_forecasting_leakage.py \
  tests/test_forecast_partitions_preprocessing.py \
  tests/test_forecasting_dataset_artifacts.py \
  tests/test_data_pipeline.py tests/test_inspect_dataset.py
```

**Result: 101 data tests passed**, including 80 new conditional-workflow tests
and 21 existing data/inspection tests. No tests that train models were run.
All eight uploaded-data/export checks passed. Hash comparisons confirmed that
all 151 pre-existing source, documentation, result, and artifact files remained
unchanged, including the complete feasibility report. The aggregate evidence
is saved in the ignored `validation_checks.json` report.

## Remaining limitations

The complete-history requirement favors participants with more continuous
wearable observation; no missingness mechanism or population generalizability
is established. The original released cohort already depended on future
endpoint survey/wearable coverage, so it is not a prospectively assembled
SM2-eligible population. Group separation does not remove those selection
effects or the dependence of quarters within a participant.

The remaining evidence gap is an authenticated export/feature-construction
timing note: historical input windows, no-look-ahead computation, survey and
device synchronization delays, and availability bounds. The papers' nominal
monthly alignment does not prove these operational facts. Removing SM3 inputs
removes known target-proximal data, but SM1/SM2 nonoverlap with the PHQ-9
two-week recall period still needs dates or reliable relative bounds. This
dataset forecasts a later questionnaire response under stated assumptions;
it does not establish a wholly future symptom interval or clinical readiness.

This step stops at conditional data construction and validation. There are no
new model results, training runs, commits, or pushes.
