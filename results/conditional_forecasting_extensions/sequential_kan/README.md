# KAN-encoded recurrent control

Frozen aggregate snapshot of the completed non-ODE control: static context
initializes a 16-wide hidden state; a shared KAN encoder and conventional
GRUCell process SM1 then SM2, followed by a KAN regression head.

Training used all 5,111 training samples from 2,025 participant groups and the
same 851 validation samples from 338 groups. Initialization seeds were
42, 43, and 44. The actual historical configuration has `gradient_clip: null`;
it must not be described as perfectly matched to the later clipped KAN-ODE.
See [the recurrent report](../../../docs/forecasting_sequential_kan.md).

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
