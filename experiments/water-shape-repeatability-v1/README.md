# Shape repeatability and correction efficiency

The [plan](../../docs/plans/water-shape-repeatability-v1/plan.md) owns this
sequence. configs/sequence.json selects reusable inputs, reference, training
configurations and physical studies. Existing conditioning assembly, normal
Tuldok workflow and multigrid evaluation own scientific execution. Historical
records remain intact; outputs are fresh and are never blindly replayed.

Training uses CUDA; fresh checkpoint restore and qualification use CPU. Capture
committed source before managed execution. Timed command records distinguish
preparation, managed execution and CPU qualification from optimizer intervals.
Concurrent CPU/GPU observations are throughput measurements, not isolated latency.
