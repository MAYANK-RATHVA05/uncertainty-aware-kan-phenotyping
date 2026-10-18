# Executed experiment report

Date: 2026-10-18. These are actual local runs on the supplied CSV, not results imported from the source paper. The study is exploratory retrospective PHQ-9 change regression.

## A. Repository changes

Implemented modular spline KAN layers, masked causal temporal reference, a separate tabular KAN, explicit preprocessing and provisional group splitting, stable focal-loss/binary-metric utilities, regression training/evaluation, baseline and ablation commands, validation interpretation exports, a content-based dataset inspector, tests, pinned dependencies, and a repeatable setup script. No raw data or generated model artifacts are included in the source tree intended for Git.

## B. Verified schema and target

The actual CSV has 35,694 rows and 155 columns. All 154 dictionary features are present; `__index_level_0__` is undocumented. See [the full schema](data_schema.md) and [data audit](dataset_audit.md). Both PHQ score endpoints are available for 10,866 rows; 24,828 rows with missing outcomes are excluded without target imputation.

The documented three-month target is `phq9_score_end - phq9_score_start`. It remains continuous. Classification class counts, AUROC/AUPRC, sigmoid probabilities, focal training, and calibration are inapplicable to this task. No clinical diagnosis or clinically meaningful-change threshold is asserted.

## C. Implemented architecture

The real-data model is **111 transformed features → KAN(64) → KAN(1)**. Every KAN edge uses a learned cubic B-spline expansion plus a weighted SiLU base; each output has one bias. There are eight fixed uniform intervals on [−3,3] and eleven basis functions. The scalar output estimates measured PHQ change; it is unbounded and has no sigmoid.

## D. Departures from the paper

The paper studies daily All of Us sequences and binary multitask outcomes. This implementation trains one tabular continuous-outcome model using PSYCHE-D aggregates. It uses MSE and validation RMSE in place of focal loss and macro-AUROC. It has no anxiety head or task weighting. The 3/5/7 temporal KAN with 22/21/21 pooled channels and a width-64 mixer is implemented and tested as a **reference only**, without a real-data temporal experiment. [The paper audit](paper_audit.md) distinguishes specified details from disclosed choices.

## E. Leakage controls and unresolved limits

The 56 admitted raw fields are 29 described Static and 27 Statistics features. All four PHQ columns and the identifier are excluded, as are 84 regression-window summaries and ten intervention-change fields with uncertain timing. Train-only median imputation, missingness indicators, scaling, and one-hot categories produce 111 inputs.

One fixed split seed (42) separates provisional composite-index prefixes: 7,567/1,116/2,183 train/validation/test rows, corresponding to 2,825/404/807 prefix groups. No prefix crosses partitions. Actual participant identity and measurement timing are not established; the user has no confirming documentation. This is **not verified participant-level or prospective validation**. Consistent static fields and adjacent endpoint scores support the grouping inference without proving its semantics.

## F. Tests and setup verification

The complete suite passed: **107 tests in 5.14 seconds**, with no failures or skips.
After improving figure tick readability, all 17 affected interpretation tests
passed again. Tests include independent spline values, gradients, padding and
causality, leakage guards, missingness, exact group separation, metric edge cases,
checkpoint restoration, seed reproducibility, input-fingerprint rejection, and
coherent categorical/missingness permutation. Synthetic tests are not empirical
PSYCHE-D results.

Setup was executed in the real instance and rerun successfully, preserving
TLS/artifact verification; dependency compatibility and imports passed. CUDA
was unavailable; runs used CPU with two threads. Writable workspace caches
resolve the machine's read-only home cache locations.

The onboarding configuration draft successfully saved **install_script** and
**start_skill**, including venv activation, writable cache exports, data locations,
readiness checks, and the existing-isolated-checkout guidance. No secret or
network-policy changes were required. This is saved configuration, not
publication or validation in a new task. Review and save the environment
settings, then publish the environment to activate its snapshot.

## G. Executed training

- Real smoke: two epochs on group-preserving subsets (716/102/204 train/validation/test rows), best epoch 2. Independent checkpoint reloading reproduced its saved validation and test metrics exactly.
- Full KAN: initialization seeds 42, 43, and 44, with identical saved split, preprocessing, target, optimizer, and maximum budget. All executed 11 epochs and selected epoch 1 using validation RMSE, then stopped after ten nonimproving epochs. The maximum allowed budget was 50 epochs.
- Optimizer: AdamW, learning rate 0.001, weight decay 0.0001, batch size 128; MSE, patience 10, gradient clipping at norm 5. No test scores selected checkpoints, thresholds, or initialization seeds.
- Fixed mean and Ridge(alpha=1) baselines, 100-tree random forest, and width-64 MLP executed on the exact saved transformed split. Neural baselines used the same 50-epoch maximum and validation stopping protocol.
- Seed-42 tabular ablations: conventional linear/SiLU first-layer mixing with a KAN head, and Huber loss instead of MSE. Both executed 11 epochs, selecting epoch 1. No temporal or focal-loss ablation was claimed.

## H. Measured validation and held-out metrics

All metrics below refer to PHQ-score-change regression; lower MAE/RMSE is better. R² near zero indicates essentially no explained variance. No held-out result is treated as confirmatory.

| Run | Validation MAE | Validation RMSE | Validation R² | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| Smoke KAN (subset) | 3.244305 | 4.384047 | -0.004498 | 3.102773 | 4.312331 | -0.007168 |
| KAN, seed 42 | 2.962156 | 4.270672 | 0.000214 | 3.002516 | 4.217428 | 0.000256 |
| mean | 2.983808 | 4.272364 | -0.000579 | 3.018504 | 4.219018 | -0.000498 |
| ridge | 3.001271 | 4.290538 | -0.009109 | 3.029715 | 4.229482 | -0.005467 |
| random_forest | 3.064444 | 4.347469 | -0.036067 | 3.064260 | 4.259290 | -0.019689 |
| mlp | 2.974788 | 4.271651 | -0.000245 | 3.012257 | 4.220973 | -0.001425 |
| mlp_mixing | 2.961464 | 4.270091 | 0.000485 | 3.005595 | 4.220145 | -0.001032 |
| huber | 2.962039 | 4.271499 | -0.000174 | 3.002731 | 4.218962 | -0.000471 |

Three KAN initialization seeds, mean ± sample standard deviation (`ddof=1`):

| Partition | MAE | RMSE | R² |
|---|---:|---:|---:|
| validation | 2.968711 ± 0.009652 | 4.271070 ± 0.001241 | 0.000027 ± 0.000581 |
| test | 3.007171 ± 0.006498 | 4.218165 ± 0.000689 | -0.000093 ± 0.000327 |

**Interpretation:** the mean KAN test RMSE is effectively the same as the constant training-mean baseline (4.219018), and mean test R² is slightly negative. These runs do not demonstrate useful prediction of change or establish KAN superiority. Small differences among models are not evidence of clinical significance. Baselines and ablations use one initialization seed; their uncertainty is not compared as if it matched the three-seed KAN estimate. Initialization spread is not a population confidence interval.

## I. Local artifacts

- Aggregate inspection: `artifacts/inspection/{structure,data_audit}.json`.
- Smoke: `runs/smoke/`.
- Full runs: `runs/full-seed42/`, `runs/full-seed43/`, `runs/full-seed44/`.
- Each full run: `best.pt`, `history.json`, `training_config.json`, `run.json`, `preprocessor.joblib`, `splits.npz`, `source_row_positions.npz`, `metrics.json`, and `evaluation_restored.json`.
- Three-seed summary: `artifacts/results/kan_three_seeds.json`.
- Fair comparisons: `runs/full-seed42/baselines/` and `runs/full-seed42/ablations/`.
- Interpretation exports: `runs/full-seed42/interpretation/`: learned spline/SiLU
  edges, conditional feature responses, and validation permutation importance,
  each as PNG/PDF and aggregate CSV/JSON, plus repeat-level importance CSV and
  the checkpoint/input-hash interpretation manifest.

The interpretation run restored the prespecified seed-42 best checkpoint and
used only its 1,116 validation rows. Five permutations covered all 56 original
features and all 111 encoded coordinates; each original feature's one-hot block
and numeric/missingness pair stayed together. The four response/edge features
were selected using validation permutation scores, never test scores. The hidden
output unit (0) was fixed in advance. Three PNGs were verified at 300 dpi,
with matching PDF and metadata exports; learned-edge and importance figures
were visually reviewed. Input axes use transformed units. Importance changes
are tiny, consistent with the model's negligible overall skill. No feature is
therefore promoted as a clinically significant predictor. Row permutations do
not preserve provisional-group dependence, and correlated features can take
perturbed inputs outside observed joint support.

The raw CSV and dictionary are retained outside the checkout at `/workspace/private-data/psyche-d/`. Source row positions and fitted artifacts stay in ignored directories. No individual participant records are printed. All three full-run restored evaluations exactly matched their original metric JSON; shared split hashes were verified across seeds, baselines, and ablations.

## J. Limitations and unrun work

Identity, observation units, calendar timing, source dataset licensing/provenance, and feature availability need authoritative documentation before temporal forecasting or clinical conclusions. Missing outcomes exclude 69.56% of rows. Selection, confounding, repeated observations, and provisional grouping limit generalization. No external cohort, clinical validation, binary classification experiment, genuine temporal experiment, or full five-seed replication was performed. No causal contribution or uncertainty-calibrated clinical system is claimed.

## K. Reproduction

Use the tested commands in [README](../README.md), starting with `bash scripts/setup_environment.sh` and activating the venv. Run `python -m pytest -q`. Reuse the data and dictionary with their saved SHA-256 fingerprints; use new empty output directories for training. Repeat initialization seeds 42/43/44 while keeping split seed 42. Saved-run evaluation verifies the input fingerprints, split boundaries, features, checkpoint architecture, and training configuration. This report covers experimental validation; publishing the cloud environment snapshot is a separate action.
