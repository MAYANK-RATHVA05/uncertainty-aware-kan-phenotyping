# Conditional relative-month forecasting results

These files are frozen aggregate snapshots of the completed dataset, baseline,
KAN, and Ridge feature-ablation steps. They preserve the results already
reported in the corresponding research documents; preparing this bundle did
not rebuild the cohort, refit preprocessing, train models, or evaluate reserved
calibration/test payloads.

The task predicts measured SM3 PHQ-9 severity from conditionally available SM2
information. Its horizon is one relative study-month interval. The export
mapping, enrollment-demographic interpretation, and availability of historical
wearables and SM0 PHQ-9 remain explicit assumptions. Actual available-at
timestamps and prospective clinical validity are unverified. See
[temporal feasibility](../../docs/temporal_feasibility.md) and the
[dataset report](../../docs/forecasting_dataset.md).

| Folder | Snapshot contents | Report |
| --- | --- | --- |
| `dataset/` | Cohort assumptions, eligibility/exclusions, aggregate split counts, tensor shapes, preprocessing metadata, and integrity checks | [Forecasting dataset](../../docs/forecasting_dataset.md) |
| `baselines/` | Persistence, training-mean, and full Ridge configuration, validation metrics, provenance, and checks | [Baselines](../../docs/forecasting_baselines.md) |
| `kan/` | Fixed three-seed KAN configuration, validation results, aggregate sanity/epoch histories, learning curves, provenance, and checks | [KAN](../../docs/forecasting_kan.md) |
| `feature_ablations/` | A–D feature lists, alpha selection, validation comparisons, full Ridge reproduction checks, and provenance | [Feature ablations](../../docs/forecasting_feature_ablations.md) |

All model comparisons use the same 851 validation samples from 338 participant
groups. Training contains 5,111 samples from 2,025 groups. Repeated quarters
within a participant are dependent, and validation also selects Ridge alpha
and KAN checkpoints. These exploratory comparisons do not establish causal
effects, statistical significance, wearable-specific benefit, or performance
on unseen test data. Initialization variation is not a population confidence
interval. Calibration and test performance were not evaluated in these steps.
Their aggregate sizes appear in dataset metadata only.

The JSON files are byte-for-byte copies of the original aggregate artifacts;
they retain their original local paths and provenance hashes. References in
`run.json`, `manifest.json`, and check files can point to private artifacts or
the original artifact directory layout. Those references do not mean that
the referenced files are included here. The original private artifacts remain
unchanged. [aggregate_manifest.json](aggregate_manifest.json) records the
repository-relative source, destination, and SHA-256 of each of the 39 copied
files. Its hashes apply to the copied file bytes. It excludes this README and
the aggregate manifest itself.

[publication_checks.json](publication_checks.json) records the final publication
verification: all 248 repository tests passed and all 39 copied artifacts match
their original SHA-256 values. It is an additional check report, not a historical
experiment artifact listed in the copy manifest.

This bundle contains only JSON aggregate metadata/results and PNG/SVG learning
curves. Raw data, participant identities and row mappings, split-membership
arrays, measured individual outcomes, aligned individual predictions,
preprocessors, fitted models, and checkpoints remain outside Git. No NPZ,
joblib, or PT files are redistributed.
