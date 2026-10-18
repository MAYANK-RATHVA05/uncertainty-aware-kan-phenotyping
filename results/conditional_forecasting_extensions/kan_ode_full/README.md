# Full latent KAN-ODE training and validation

Frozen aggregate snapshot of fresh initialization seeds 42, 43, and 44 using
all 5,111 saved training samples from 2,025 participant groups and the same
851 validation samples from 338 groups. No smoke or overfit weights are reused.

The implemented architecture, 0.25 RK4 step size, 0.01 dynamics initialization
scale, AdamW settings, 50-epoch limit, and patience 10 remain fixed. Gradient
norm clipping is 5; the historical non-ODE recurrent control had no clipping.
Step-halving comparisons use fixed training diagnostic inputs, never validation
accuracy to choose a solver. Initialization spread is sample SD across seeds,
not population uncertainty.
See [the full results report](../../../docs/forecasting_kan_ode_results.md).

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
