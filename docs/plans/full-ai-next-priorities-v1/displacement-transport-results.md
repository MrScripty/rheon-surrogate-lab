# Face and cell transport representation qualification

CPU and CUDA each pass all 80 registered cases: 64 captured training windows and 16 manufactured fields, covering four domain shapes and both spatial densities. The training-only tensor labels match native face/cell displacement and point-velocity labels exactly. Given native displacements, the fixed readout reproduces native velocity and tracer outputs exactly at float32 precision.

Gradients with respect to fields and displacements are finite and bitwise repeatable on each device. Zero and constant tracer behavior also passes. The captured next-tracer fields match the native primitive bitwise.

These checks qualify labels and a representation, with zero optimizer updates. They do not establish a learned forecast, full-state rollout accuracy or CPU latency. A candidate must predict its own displacements from causal inputs; native backtracing and reference fields remain training-only. Every requested horizon and the complete 4/6/8 ms block budgets still apply.

[Evidence and source hashes](displacement-transport-results.json) · [Frozen protocol](displacement-transport-qualification-protocol-01.json) · [Timing and architecture definitions](multi-step-generation-options.md)
