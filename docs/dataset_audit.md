# PSYCHE-D data and task audit

## Files and verified observations

The usable upload is `anon_processed_df_parquet.zip` (24,974,557 bytes). Its
single member, `anon_processed_df_parquet.csv`, is **CSV text**, not Parquet.
The extracted file is 61,016,817 bytes, with SHA-256
`3d96afaa768bcd50c56524b931ac2e6acdd5f08c95db16c1e3b13d7dd6ce3934`.
No file extension was changed to make a reader accept it.

The CSV has **35,694 rows and 155 columns**. The supplied feature dictionary
documents 154 distinct names, all present in the CSV; its only undocumented
column is `__index_level_0__`. There are 861,072 missing cells, no nonfinite
numeric values besides missingness, and no duplicate complete rows after
excluding the index. The [complete schema](data_schema.md) lists inferred CSV
types and missing counts for every column. Aggregate machine-readable details
are saved locally in `artifacts/inspection/data_audit.json`.

The dictionary groups 154 fields into 65 Sleep, 46 Steps, 39 Demographic,
and 4 PHQ-9 fields. Its subcategories are 84 Linear regression, 29 Static,
27 Statistics, 10 Dynamic, 2 PHQ-9 score, and 2 PHQ-9 category fields.

## Target verified against the dictionary

| Actual column | Documented meaning | Observed values | Missing rows |
|---|---|---|---:|
| `phq9_score_start` | PHQ-9 score at start of the three-month period | 0–27 | 24,828 |
| `phq9_score_end` | PHQ-9 score at end of the three-month period | 0–27 | 24,828 |
| `phq9_cat_start` | PHQ-9 category at start of the three-month period | 0–4 | 24,828 |
| `phq9_cat_end` | PHQ-9 category at end of the three-month period | 0–4 | 24,828 |

The four fields share their missingness pattern. **10,866 observations** have
both score endpoints. Their continuous target is

`change = phq9_score_end - phq9_score_start`.

Positive change denotes a higher measured symptom score, negative change a
lower score. This is measured severity change, not a diagnosis, a screening
status, or a binary clinically meaningful-change threshold. The category codes
are not substituted for an undocumented classification definition.

Observed change ranges from −24 to 27, with mean −0.1943 and sample standard
deviation 4.1770. There are 4,712 decreasing, 1,866 unchanged, and 4,288
increasing observations. These descriptive counts are not classification
labels. Binary class balance, AUROC, AUPRC, focal loss, Brier score, and
probability calibration are not applicable to the selected regression task.

## Observation unit, identity, and time remain partly unresolved

All index values consist of two integer tokens separated by an underscore.
There are 35,694 unique complete keys, 4,948 distinct prefixes, and suffixes
1 through 12. Labels occur only at suffixes 3, 6, 9, and 12; the labeled
subset contains 4,036 prefixes. This is consistent with participant-month
observations and quarterly assessments, but the dictionary does not establish
those meanings. The user confirmed that no supporting index/timing
documentation is available. Neither token is treated as a verified timestamp.

Two additional aggregate checks support, but do not document, the proposed
grouping: all 29 Static fields are constant within each prefix when nonmissing,
and all 6,471 consecutive labeled suffix pairs separated by three have a
previous end score equal to the next start score. These checks do not establish
calendar units, measurement cutoffs, or future data availability.

The current analysis therefore uses **indexed aggregate observations**. It
does not turn suffix order into temporal sequences, claim that arbitrary rows
are independent participants, or apply kernels to ordered feature columns.
Whether valid sequences of length 3, 5, or 7 exist remains unverified.

Splits keep each observed index prefix together as a **provisional grouping
rule**, using reference group proportions 70% / 10% / 20%. This prevents the
same prefix crossing splits. It cannot establish separation of real people
if multiple prefixes refer to one person. Split metadata and reports explicitly
retain that limitation. Regression splitting is not class-stratified.

## Conservative predictors and leakage boundaries

The tabular adaptation admits the 29 described Static and 27 described
Statistics fields. It excludes:

- All four PHQ-9 fields, including start scores and categories, because they
  directly define or encode the target.
- The index and any undocumented columns, which are never model features.
- All 84 Linear regression summary fields: their descriptions and aggregation
  windows are absent from the supplied dictionary.
- All 10 Dynamic intervention/change fields: their availability relative to
  the outcome window is unverified.

The complete exclusion map is saved with each fitted preprocessor. Named
categorical fields are encoded explicitly; numeric/binary features are handled
separately. Imputation, scaling, one-hot categories, and missingness indicators
are learned from training groups only. Held-out groups never refit them.
All baselines and ablations reuse the same saved split and fitted transformations.

The feature dictionary does not establish exact measurement times even for
the included aggregate fields. Results are therefore **exploratory retrospective
associations with measured PHQ-9 change**. They do not demonstrate future
prediction, absence of all temporal leakage, clinical utility, or diagnosis.
No causal interpretation is warranted. Labels missing from 69.56% of rows,
unverified grouping, confounding, repeated observations, and sample selection
limit generalization.

## Requirements for a temporal forecasting study

Before activating forecasting, obtain authoritative documentation of the index,
observation intervals, aggregation windows, PHQ assessment dates, and predictor
availability at a defined cutoff. Then check gaps, duplicates, history lengths,
target alignment, static-covariate measurement times, and group identity.
The separately implemented temporal model remains a tested reference module;
it has not been validated on genuine PSYCHE-D sequences. A guarded forecasting
preparation helper does not supply the missing scientific evidence.
