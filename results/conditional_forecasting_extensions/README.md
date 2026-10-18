# Conditional forecasting extensions: public aggregate results

These snapshots accompany the sequential KAN control, latent KAN-ODE
implementation, and full latent KAN-ODE experiment. They preserve earlier
results and retain their conditional relative-month interpretation.

| Folder | Scope |
| --- | --- |
| [sequential_kan/](sequential_kan/) | Completed non-ODE recurrent control, seeds 42–44 |
| [kan_ode_implementation/](kan_ode_implementation/) | Numerical checks, separate overfit check, and preliminary two-epoch smoke run |
| [kan_ode_full/](kan_ode_full/) | Fresh full-training KAN-ODE seeds 42–44, diagnostics, and validation comparisons |

Full experiments use the same 5,111 training samples from 2,025 participant
groups and 851 validation samples from 338 groups. The earlier smoke run
uses 340 training samples from 128 groups and is labeled separately. The
historical recurrent control has no gradient clipping; latent KAN-ODE uses
norm clipping at 5. Their configurations are therefore not perfectly matched.

The task predicts measured SM3 PHQ-9 severity using measured SM0, enrollment
demographics, and wearable packets at SM1 and SM2. Timing and export mapping
remain conditional assumptions; the horizon is one relative study-month
interval, not 30 days. No measured clinical latent trajectory, causal benefit,
KAN-specific advantage, or methodological novelty follows from these results.
Repeated quarters are dependent. Initialization SD is not a population
confidence interval. No calibration/test payloads were used.

Only aggregate JSON and learning curves are included. Private checkpoints,
aligned predictions and measured individual outcomes, participant/sample
identifiers, split memberships, fitted preprocessing, and raw data stay local
and ignored by Git. References to their filenames/hashes provide provenance
without publishing their contents. Absolute private paths are redacted;
repository paths are relative. Section manifests record both original and
published SHA-256 values plus path-only transformations. All original private
artifacts and earlier tracked publication snapshots remain unchanged.

The root [aggregate_manifest.json](aggregate_manifest.json) hashes every file
in this bundle except itself. [publication_checks.json](publication_checks.json)
records strict format/path/field and hash checks. These publication checks do
not substitute for numerical or scientific validation; original experiment
checks remain in their corresponding sections.
