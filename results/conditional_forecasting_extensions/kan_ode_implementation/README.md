# Latent KAN-ODE implementation and smoke check

Frozen aggregate snapshot of implementation checks, the separate training-only
overfit check, and the original two-epoch seed-42 smoke run. The smoke cohort
contains 340 training samples from 128 training participants, retaining all of
their saved samples. Validation uses the existing 851 samples from 338 groups.
These preliminary metrics are not a full-training comparison.

The 26,257-parameter latent model adds a shared autonomous KAN dynamics field
to the recurrent architecture and integrates relative study-month intervals
0→1, 1→2, and 2→3 using classical fixed-step RK4 at 0.25. Observation updates
use the two available historical packets only. Dynamics base/spline weights
start at 0.01 times their original initialization; bias starts at zero.
Gradient norm clipping is 5.
See [the implementation report](../../../docs/forecasting_kan_ode_implementation.md).

The target is measured SM3 PHQ-9 severity from conditionally available SM2
information, with one relative study-month interval as the horizon. Exact
available-at timestamps, target recall-window separation, and prospective
clinical validity remain unverified. Learned hidden coordinates are not
measured clinical states. Calibration/test payloads remain unused.

JSON snapshots preserve aggregate values while making repository-local paths
relative and redacting external private paths. `aggregate_manifest.json`
records original and published hashes and each path transformation. Historical
checks describe their original run, including its then-current Git state.
Checkpoint weights, aligned individual predictions, split/index arrays, raw
data, and fitted preprocessing remain local and ignored by Git. Filenames and
SHA-256 references to those excluded files are provenance, not their contents.
