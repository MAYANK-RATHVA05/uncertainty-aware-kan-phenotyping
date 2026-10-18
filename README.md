# Depression-only KAN research with PSYCHE-D

This project adapts the spline-based layers from **Interpretable-MTLNet: A
Kolmogorov–Arnold Network for Multitask Mental Health Prediction**, by Mai Ali,
Zhenghao Ni, and Deepa Kundur, to one depression-related outcome. It does not
implement anxiety prediction, causal modelling, or clinical diagnosis, and does
not reproduce the paper's All of Us experiments.

The implemented PSYCHE-D study is **exploratory retrospective tabular regression**
of `phq9_score_end - phq9_score_start`, using documented three-month PHQ-9
measurements. The scores remain continuous; no binary screening threshold is
invented. See the [dataset audit](docs/dataset_audit.md),
[complete schema and missingness](docs/data_schema.md), and
[paper audit](docs/paper_audit.md).

## What the available data supports

The actual uploaded file is CSV inside a ZIP, despite the word `parquet` in
its filename: 35,694 rows, 155 columns, and 10,866 labeled rows. All 154
dictionary features are present. The index looks like two integer tokens,
but its participant/month meanings and feature measurement windows are
unconfirmed; the user has no documentation establishing them.

Consequently, the running study uses a tabular KAN and keeps matching index
prefixes together as **provisional groups**. This does not establish real
participant separation. No temporal sequences are constructed from those
tokens. Results measure association with the observed PHQ-9 change, not
future forecasting performance. Missing labels, repeated observations,
confounding, and sample selection further limit the findings.

The repository also contains a separately tested temporal KAN reference
implementation with kernels 3/5/7. It remains **untrained on PSYCHE-D**.
Temporal or prospective claims require authoritative identity, assessment-date,
and feature-window documentation first.

## Environment and private inputs

The tested cloud environment uses Linux, Python 3.12.14, uv 0.12.23, and
PyTorch 2.9.1 CPU. Python 3.12 is required. No server, credentials, restricted
All of Us dataset, or external dataset download is needed.

```bash
cd /workspace/uncertainty-aware-kan-phenotyping
bash scripts/setup_environment.sh
source /workspace/.venvs/uncertainty-aware-kan-phenotyping/bin/activate
export MPLCONFIGDIR=/workspace/.cache/matplotlib
export XDG_CACHE_HOME=/workspace/.cache
export MPLBACKEND=Agg
```

The setup script installs the CPU wheel from PyTorch's official index, the
other exact versions from `requirements.lock.txt`, and this project in editable
mode. It checks dependency compatibility and imports. It can be run again.
TLS and artifact verification remain enabled. For another checkout location,
set `KAN_VENV_DIR`, `UV_CACHE_DIR`, and `MPLCONFIGDIR` to writable local paths.
Use the existing isolated task checkout; a separate Git worktree is unnecessary.

The user-supplied dataset is described as PSYCHE-D. Dataset provenance and
reuse permissions must come from its original distribution; the upload and
dictionary do not establish a license or a publication identifier. The paper
and data are not redistributed in this repository. Place private copies at:

```text
/workspace/private-data/psyche-d/anon_processed_df_parquet.csv
/workspace/private-data/psyche-d/feature_explanations.tsv
```

Those locations are already prepared in the current cloud instance. Future
users must supply their own authorized copies if absent. Keep raw data,
participant identifiers, model checkpoints, and generated outputs out of Git.
`.gitignore` covers their usual formats and output directories.

## Inspect before modelling

```bash
python tools/inspect_dataset.py \
  /workspace/private-data/psyche-d/anon_processed_df_parquet.csv \
  --output artifacts/inspection/structure.json
```

The inspector detects CSV versus Parquet from contents, reports aggregate
schema/missingness, and does not print participant records. Name-based target
candidates are explicitly unverified. The project-specific audit reconciles
the actual columns against the dictionary; it does not infer a target from a
suggestive name alone.

## Data handling and leakage limits

The reference split seed is 42, separate from model initialization seeds.
The actual split contains 7,567 / 1,116 / 2,183 observations and
2,825 / 404 / 807 provisional prefix groups. Group proportions approximate
70% / 10% / 20%; row proportions differ because group sizes vary. Continuous
regression is not class-stratified. No prefix crosses splits.

Only 29 described Static and 27 described Statistics fields are admitted.
All PHQ fields and identifiers are excluded; 84 Linear regression fields with
undocumented windows and 10 Dynamic intervention fields with uncertain timing
are also excluded. The saved preprocessor records every exclusion reason.

Numeric/binary features receive training-only median imputation, missingness
indicators where training missingness occurs, and training-only standardization.
`sex`, `educ`, `money`, `household`, and `num_migraine_days` are explicitly
categorical: missing category, training-only one-hot categories, and unknown
held-out levels ignored. This avoids inventing the undocumented migraine
category mapping. The full split yields 111 transformed predictors.

Even after these exclusions, predictor availability relative to the PHQ
outcome period cannot be verified. This is an explicit limitation, not a claim
that all temporal leakage has been eliminated.

## Architecture and disclosed choices

Each KAN layer computes

\[
y_j=b_j+\sum_i\left[\sum_r c_{jir} B_{r,p}(x_i)+w_{ji}\operatorname{SiLU}(x_i)\right].
\]

The actual tabular model is `111 -> KAN(64) -> KAN(1)` and returns an
unbounded scalar PHQ-9 change estimate. It does not apply sigmoid. The
following choices are explicit adaptations; the paper does not specify them:

- Cubic splines (`p=3`), eight uniform intervals on `[-3,3]`, knot extensions
  at both ends, and eleven basis functions per input edge.
- Cox–de Boor basis evaluation with fixed knots. The finite spline component
  tapers outside its support; the SiLU base continues to extrapolate. No
  adaptive-grid or exact-paper-equivalence claim is made.
- Kaiming-uniform base weights, small random spline weights, output bias once
  after edge summation, and no additional activation between KAN layers.

The temporal reference uses 22/21/21 channels across kernels 3/5/7, for **64
total**; causal left padding that masks absent edge contributions; temporal
average pooling; concatenation; a KAN mixing layer of width 64; and one
scalar depression head with optional static covariates concatenated once.
Channel allocation, padding, and exact head dimensions are disclosed choices.
Static fields are not duplicated in temporal inputs. These modules do not
convert tabular columns into time steps.

## Train, restore, and evaluate

```bash
python -m kan_phenotyping.experiments.train_depression \
  --data /workspace/private-data/psyche-d/anon_processed_df_parquet.csv \
  --dictionary /workspace/private-data/psyche-d/feature_explanations.tsv \
  --config configs/depression_kan.yaml --output runs/my-smoke \
  --max-epochs 2 --max-samples 1024

python -m kan_phenotyping.experiments.train_depression \
  --data /workspace/private-data/psyche-d/anon_processed_df_parquet.csv \
  --dictionary /workspace/private-data/psyche-d/feature_explanations.tsv \
  --config configs/depression_kan.yaml --output runs/my-full-seed42 --seed 42

python -m kan_phenotyping.experiments.evaluate_depression \
  --run-dir runs/my-full-seed42 \
  --data /workspace/private-data/psyche-d/anon_processed_df_parquet.csv \
  --dictionary /workspace/private-data/psyche-d/feature_explanations.tsv
```

Choose a new empty output directory for each run. The smoke cap preserves
whole groups within each existing partition. For multiple initialization
seeds, use `--seed 42`, `43`, and `44`, while retaining `split_seed: 42`.

AdamW uses learning rate `1e-3`, weight decay `1e-4`, batch size 128, maximum
50 epochs, and patience 10. The selected regression task uses MSE and selects
the minimum **validation RMSE**, then restores that checkpoint before held-out
test evaluation. Gradient norm clipping at 5 and two CPU threads are additional
engineering choices. This replaces the paper's classification focal loss and
macro-AUROC. Standalone stable focal-loss and binary-metric utilities are tested
for callers with an independently justified binary task; the PSYCHE-D CLI never
silently binarizes PHQ scores.

Runs save input hashes, exact split indices, source row positions, fitted
preprocessing, feature/exclusion metadata, configuration, dependency versions,
best checkpoint, epoch loss/validation history, and separate validation/test
metrics. Regression results are MAE, RMSE, and R² in PHQ-score-change units.
Saved-run evaluation checks data/dictionary hashes before restoring artifacts.
The test set does not select checkpoints, hyperparameters, thresholds, or
which initialization seed receives interpretation plots.

## Comparisons, explanations, and tests

A training-mean predictor, ridge regression, a random forest regressor, and an MLP provide
task-compatible baselines; logistic regression is inapplicable to this continuous
target. Comparisons reuse saved partitions and transformations. Temporal
CNN/LSTM comparisons and temporal ablations remain unrun because chronology
has not been verified. KAN superiority is not assumed.

```bash
python -m kan_phenotyping.experiments.run_baselines \
  --run-dir runs/my-full-seed42 \
  --data /workspace/private-data/psyche-d/anon_processed_df_parquet.csv \
  --dictionary /workspace/private-data/psyche-d/feature_explanations.tsv

python -m kan_phenotyping.experiments.run_ablations \
  --run-dir runs/my-full-seed42 \
  --data /workspace/private-data/psyche-d/anon_processed_df_parquet.csv \
  --dictionary /workspace/private-data/psyche-d/feature_explanations.tsv

python -m kan_phenotyping.experiments.summarize_runs \
  --runs runs/full-seed42 runs/full-seed43 runs/full-seed44 \
  --output artifacts/results/kan_three_seeds.json

python -m kan_phenotyping.experiments.interpret_depression \
  --run-dir runs/my-full-seed42 \
  --data /workspace/private-data/psyche-d/anon_processed_df_parquet.csv \
  --dictionary /workspace/private-data/psyche-d/feature_explanations.tsv \
  --n-repeats 5
```

Tabular ablations replace the first KAN mixer with a conventional linear/SiLU
map, or replace MSE with Huber loss. The target, split, and maximum epoch budget
otherwise stay fixed. The losses have different raw scales; both select
checkpoints using validation RMSE.

Interpretation exports selected spline-plus-SiLU edges, conditional feature
responses, and separately identified validation permutation importance.
Permutation jointly shuffles all encoded coordinates of one original feature,
preserving complete one-hot vectors and numeric value/missingness pairs.
Transformed feature names and preprocessing metadata accompany the exports.
Curves and importance show model associations; prominent features are not
proof of clinical significance or causality. Validation rows may be dependent
within provisional groups, so row-level permutation rankings are descriptive.

```bash
python -m pytest -q
```

Tests cover spline values/gradients, temporal shapes/padding/causality, loss
numerics, group separation, preprocessing leakage and missing values, strict
metrics, checkpoint restoration, and interpretation exports. Synthetic fixtures
are identified as such; their checks are not PSYCHE-D experiment results.
Actual executed experiment results and artifact paths are recorded in
[the experiment report](docs/experiment_report.md).
