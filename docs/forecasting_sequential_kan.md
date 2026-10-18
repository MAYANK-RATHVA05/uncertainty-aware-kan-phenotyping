# KAN-encoded recurrent forecasting baseline, without ODE

Run on **2026-10-18**, following the [dataset](forecasting_dataset.md),
[baseline](forecasting_baselines.md), [tabular KAN](forecasting_kan.md), and
[feature-ablation](forecasting_feature_ablations.md) reports.

**Finding:** the sequential baseline's validation RMSE is
**3.8264 ± 0.0125** across initialization seeds. This is lower than the previous
tabular KAN's 3.9120, but higher than Ridge B's 3.7915 and full Ridge D's
3.8136. Temporal modeling was not assumed to improve performance, and no
settings were changed in response to these results.

This establishes an observation-by-observation, non-ODE control for possible
later KAN-ODE work. The research sequence remains **KAN → validated future
forecasting → KAN-ODE → conformal prediction**; availability timing is still
conditional. No ODE or conformal method is implemented here.

## Architecture in simple terms

The model first summarizes enrollment demographics and the quarter's measured
SM0 PHQ-9 score into a 16-value initial memory. It reads the SM1 wearable
packet, updates that memory, then reads SM2 and updates it again. The final
memory produces a raw SM3 severity estimate.

This is a **KAN-encoded recurrent baseline with conventional GRU updates**.
It is not KAN-ODE or a replication of the paper's multi-scale convolution.
The two packet encodings share exactly the same learned parameters, as do
the two GRU updates. Static context initializes the state **once**; it is
not an extra time step. Every sample starts from its own context, with no
hidden-state carry between participants, quarters, batches, or forward calls.

For context `c` and packet `w_t`:

```text
h0 = static_KAN(c)                      63 → 16
e1 = shared_packet_KAN(SM1)             48 → 16
h1 = shared_GRUCell(e1, h0)             16 → 16
e2 = shared_packet_KAN(SM2)             48 → 16
h2 = shared_GRUCell(e2, h1)             16 → 16
y  = KAN_head(h2)                      16 → 1
```

The three KAN layers reuse the existing fixed-grid spline implementation:
eight grid intervals, cubic splines, 11 basis functions per input edge,
and grid range `[-3, 3]`, with SiLU base terms and no adaptive grid or input
clipping. The GRU has its ordinary sigmoid/tanh gates. The final severity
output is **unbounded**, without a sigmoid, clipping, rounding, or conversion
to a category.

| Component | Trainable parameters |
|---|---:|
| Static KAN, 63 → 16 | 12,112 |
| Shared wearable KAN, 48 → 16 | 9,232 |
| Shared GRUCell, input/hidden width 16 | 1,632 |
| KAN head, 16 → 1 | 193 |
| **Total** | **23,169** |

## Frozen inputs, target, and chronology

The exact saved cohort, participant assignments, preprocessing, and measured
SM3 target are reused from
`artifacts/conditional_forecasting/sm2-seed42/`. No cohort is rebuilt and no
preprocessing is fitted or applied again. The existing validated loader
checks the saved hashes, source provenance, column order, and split integrity.

| Consumed partition | Samples | Participant prefixes under the release mapping | Wearables | Static context | Target |
|---|---:|---:|---|---|---|
| Training | 5,111 | 2,025 | `[5111, 2, 48]` | `[5111, 63]` | `[5111]` |
| Validation | 851 | 338 | `[851, 2, 48]` | `[851, 63]` | `[851]` |

The sequence contains exactly **SM1 then SM2**. Each packet has 27 allowed
wearable Statistics values and 21 training-derived missingness indicators.
Static context contains the 62 transformed enrollment demographic columns
and one standardized measured SM0 score. These are the same 159 transformed
values used by the full Ridge and tabular KAN, restored to their original
float32 representation and arranged into their saved blocks.

The runner verifies that reflattening these blocks exactly reproduces the
original saved matrix. Sample indices, measured target vectors, and
participant assignments remain aligned. The target is **measured SM3 PHQ-9
severity**, not PHQ change or a generated label.

The cutoff is conditionally **SM2**, with a horizon of **one relative
study-month interval**, not 30 days. There are no fake daily observations,
interpolated states, SM3 sensing inputs, or added timestamps. The first score
is each quarter's paired measured SM0 score, not necessarily its enrollment
assessment. Demographics remain the enrollment measurements.

Calibration/test input and outcome payloads are never opened or hashed.
Only training/validation observations and shared manifest, mapping, split,
and fitted-preprocessor metadata are consumed. Raw input hashes are inherited
from the dataset manifest; the full CSV is not reopened.

## Training-only sanity check and fixed training

A separate model first overfits 32 deterministically selected **training**
samples, using seed 2026, full-batch AdamW, learning rate 0.001, weight decay
0.0001, and MSE. The predeclared diagnostic cap is 1,000 steps, stopping when
RMSE ≤0.25 and MSE ≤1% of initial MSE, with changed parameters and finite,
nonzero gradients. Validation is not an input to this diagnostic.

| Sanity result | Value |
|---|---:|
| Initial training MSE | 57.445141 |
| Final training MSE | 0.061791 |
| Final training RMSE | 0.248578 |
| Steps | 226 |
| Diagnostic optimization time | 1.397 seconds |

**Passed.** All 13 trainable parameter tensors received finite/nonzero
gradients. Its checkpoint, curve, indices, and result are saved separately
in `sanity/`. The diagnostic model and optimizer are discarded; no experiment
is warm-started from them. This check establishes optimization, not forecasting
generalization.

All three experiments use the requested fixed settings:

| Setting | Value |
|---|---|
| Initialization seeds | 42, 43, 44 |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 |
| Batch size | 128, retaining the final partial batch |
| Loss | MSE on untransformed measured SM3 severity |
| Maximum epochs / patience | 50 / 10 |
| Checkpoint rule | Lowest validation RMSE, minimum improvement 0; restore before evaluation |
| Minibatch-order seed | 42 for every initialization |
| Device / precision | CPU, two PyTorch threads, float32, deterministic algorithms |
| Gradient clipping / scheduling | None / none; nonfinite-gradient hooks fail immediately |

The unchanged existing trainer supplies optimization and checkpoint selection.
Its sequence-plus-static argument dispatch is used with the new recurrent
model; that dispatch does not add a convolution. Settings are fixed across
seeds, and there is no tuning search.

## Validation results and descriptive comparison

MAE and RMSE are in **PHQ-9 points**; R² is dimensionless. All comparisons use
the same 851 validation examples with equal weight per quarter sample.

| Sequential initialization | MAE | RMSE | R² | Selected epoch | Epochs run | Fit wall time |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2.8827 | 3.8228 | 0.6180 | 15 | 25 | 10.343 s |
| 43 | 2.8351 | 3.8404 | 0.6144 | 17 | 27 | 7.825 s |
| 44 | 2.8372 | 3.8162 | 0.6193 | 22 | 32 | 10.674 s |
| **Mean ± sample SD** | **2.8517 ± 0.0269** | **3.8264 ± 0.0125** | **0.6172 ± 0.0025** | — | 84 total | **28.842 s total** |

Fit times include training, per-epoch validation, checkpoint/history writing,
and the trainer's restoration. They exclude sanity checking, dataset loading,
independent restoration, prefix checks, and plotting. These are local CPU
wall times, not hardware benchmarks. Learning curves are saved as PNG and
SVG; training values are online epoch-average MSE shown as RMSE, whereas
validation evaluates the frozen end-of-epoch model.

| Existing comparison | MAE | RMSE | R² |
|---|---:|---:|---:|
| Persistence | 2.8496 | 4.0352 | 0.5743 |
| Ridge B: SM0 + demographics | **2.8087** | **3.7915** | **0.6242** |
| Ridge D: SM0 + demographics + SM1/SM2 | 2.8209 | 3.8136 | 0.6198 |
| Tabular KAN mean ± sample SD | 2.9576 ± 0.0075 | 3.9120 ± 0.0038 | 0.5999 ± 0.0008 |
| Sequential KAN mean ± sample SD | 2.8517 ± 0.0269 | 3.8264 ± 0.0125 | 0.6172 ± 0.0025 |

The recurrent model has lower validation error than the earlier tabular KAN,
but neither Ridge variant is beaten by any recurrent initialization on these
metrics. Relative to persistence, RMSE is lower while mean MAE is slightly
higher. Earlier comparisons are read from their unchanged saved artifacts;
no baseline or tabular KAN is refitted or retuned.

Sample SD uses **`ddof=1` across the three initializations** on one fixed split.
It is not a population confidence interval, patient-level uncertainty, or
evidence of statistical significance. No initialization is selected as an
overall winner or ensembled.

## Computational order and isolation checks

Both synthetic tests and checks on every restored experimental model establish:

- The static encoder is called once, followed by exactly
  `encode SM1 → GRU SM1 → encode SM2 → GRU SM2`.
- Both packet calls use the same KAN encoder, and both updates use the same
  GRUCell. No second encoder or recurrent parameter set is created.
- Changing only SM2 leaves `h0` and `h1` **bitwise unchanged**.
- Autograd gives **exactly zero** gradient of the SM1 state with respect to
  SM2, while showing nonzero gradients with respect to SM1 and context.
- Calling `step(SM1, initial_state(context))`, without supplying SM2 at all,
  exactly reproduces the SM1 state from the complete sequence.
- Each checkpoint loaded into a fresh model reproduces saved validation
  predictions and metrics exactly, with unchanged sample/target alignment.

No requirement or test asserts that reversing packets must worsen prediction
performance. Computation follows the intended order independently of whether
the model learns useful order-dependent information. These checks concern
the transformed-input computation; they do **not** certify the export's
actual available-at times or resolve the data-provenance assumptions.

## Artifacts, reproduction, and preservation

New files are `models/sequential_kan.py` and
`experiments/forecasting_sequential_kan.py` under `src/kan_phenotyping/`, plus
`tests/test_forecasting_sequential_kan.py` and this report. Existing files are
unchanged.

The ignored output directory is
`artifacts/conditional_forecasting_sequential_kan/sm2-fixed-seeds42-44/`:

- `config.json`, `comparisons.json`, `summary.json`, and `run.json`: frozen
  protocol/feature names, all seeds, unchanged comparison results, model and
  training configuration, timings, input/source/artifact hashes, and versions.
- `sanity/`: diagnostic checkpoint, training subset indices, history, and
  pass evidence, separate from every experiment.
- `seed_42/`, `seed_43/`, `seed_44/`: selected `best.pt`, training settings,
  learning history, validation metrics/prefix checks, and aligned
  `validation_predictions.npz` with original sample indices and measured targets.
- `preprocessor.joblib` and `used_split_indices.npz`: exact saved transform
  copy and original training/validation indices.
- `learning_curves.png` / `learning_curves.svg`: all three learning curves.
- `access_checks.json` / `validation_checks.json`: aggregate execution and
  independent verification evidence.

Using the configured Python environment and a **fresh** output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/workspace/.cache/matplotlib XDG_CACHE_HOME=/workspace/.cache \
python -m kan_phenotyping.experiments.forecasting_sequential_kan \
  --dataset-dir artifacts/conditional_forecasting/sm2-seed42 \
  --baseline-dir artifacts/conditional_forecasting_baselines/sm2-seed42 \
  --ablation-dir artifacts/conditional_forecasting_feature_ablations/sm2-seed42 \
  --tabular-kan-dir artifacts/conditional_forecasting_kan/sm2-fixed-seeds42-44 \
  --output artifacts/conditional_forecasting_sequential_kan/sm2-fixed-seeds42-44
```

**All 287 repository tests passed**, including 39 new synthetic sequential
tests. They cover the architecture, sharing, chronology, prefix isolation,
finite gradients, fixed settings, training-only fitting despite changed
validation outcomes, deterministic repeats/restoration, input reconstruction,
preprocessing guards, output shape, and raw unbounded predictions.
**Thirteen independent actual-run checks passed**, including fresh-process
restoration, computational prefix isolation for all three trained models,
unchanged preprocessing, and comparison alignment. All **106 existing tracked
files and 184 prior artifact files** remain unchanged. Runtime guards recorded
zero calibration/test payload opens during training and independent checking.
Verification scripts and file-access guards are retained outside Git at
`/workspace/scratch/forecasting_sequential_kan_validation/`.

## Limits

The four original conditional assumptions remain: release/export index
correspondence, historical summaries available by SM2 without future backfill,
earlier measured SM0 availability, and enrollment demographics without later
replacement. Actual survey/device delays, feature construction provenance,
and separation from the target's two-week recall period are unverified.
This is not prospective clinical validation or a certified wholly future
symptom interval.

Repeated quarter observations within a participant are dependent. The fixed
split prevents participant overlap but pooled metrics weight participants
with more eligible quarters more heavily. Complete-history selection and the
release's future-endpoint coverage filter may bias the cohort. Validation
checkpoint selection and repeated model comparisons create optimism; test
generalization remains unknown.

There are only two observed summary packets. A recurrent update is not
evidence of continuous physiology or proof that finer temporal structure has
been recovered. The tabular and recurrent models differ in architecture and
parameter count, so their performance difference does not isolate order or
wearable-specific benefit. A useful nonlinear/temporal signal is neither
proved nor ruled out by this experiment.

All previous results are preserved. No calibration/test payloads were used,
no ODE/conformal method was implemented, and no commit or push was made.
Work stops after this fixed sequential baseline.
