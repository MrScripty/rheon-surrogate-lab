# Primary sources for the diagnosis and next decision

Retrieved2026-10-14. These sources frame interpretation; none establishes the
cause or benefit of this exact trained model. No paper benchmark gains are reused.

| Source | Supported point | Local consequence |
| --- | --- | --- |
| [PyTorch2.8 SiLU](https://docs.pytorch.org/docs/2.8/generated/torch.nn.SiLU.html) | SiLU is x times logistic sigmoid | Inspect preactivation sign/magnitude and derivative transmission; small negative-tail output is possible, but saturation here must be measured |
| [PyTorch2.8 numerical accuracy](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html) | f32 has limited precision; mathematically identical batching/evaluation can produce different results | Match original batch2 validation and original witness path; precision-copy probes are arithmetic evidence, not restored-model parity or a new solver |
| [PyTorch2.8 GroupNorm](https://docs.pytorch.org/docs/2.8/generated/torch.nn.GroupNorm.html) / [Wu and He](https://arxiv.org/abs/1803.08494) | Describes within-sample group normalization, independent of batch statistics | A non-affine fixed feature normalization is one proposed amplitude-control intervention; vision outcomes predict no pressure benefit |
| [Wang, Teng and Perdikaris](https://arxiv.org/abs/2001.04536) | Analyzes gradient imbalance/stiffness in composite physics-informed neural losses | Measure pressure versus residual gradients separately; the PINN setting differs from this discrete pressure surrogate, so no causal diagnosis transfers automatically |
| [Tompson etal.](https://arxiv.org/html/1607.03597v7) | Pressure projection uses multiresolution context and physics losses; discusses supervised/rollout data mismatch | Keep box-spanning representation, but do not infer long rollout quality from development snapshot loss |
| [Hsiehetal.](https://arxiv.org/abs/1906.01200) | Studies learned updates coupled to existing solvers with convergence guarantees for its specific construction | Retain native residual-controlled verification and fallback; do not transfer the paper theorem to this unconstrained warm-start surrogate |

A raw1×1 head is mathematically u_i=W h_i+b. The existing gauge computes
s(u_i-u_0)=s W(h_i-h_0), so a common head bias has no exact-arithmetic scientific
signal. In float32, adding b can round small spatial differences away before the
later float64 gauge subtraction. This is a local algebraic inference; whether it
explains observed outputs, and whether feature variation was already tiny, are
separate measured questions. Any higher-precision or centered-feature probe must
first match the production float32 replay and retain unchanged stored weights.

A follow-on proposal will compare only a few practical changes against the actual
measurement: a gauge-aware output representation if rounding is material, one
fixed loss/scale ablation if gradient imbalance is demonstrated, or a simpler
global basis model if the evidence supports abandoning the current representation.
The sealed-box basis already has an exact analytic competitor and prior learned
modal evidence, so repeating it cannot be portrayed as unexplored solver research
or an expected accelerator. Selection and numeric configuration remain proposals
until parent review; no training is launched by this diagnosis.
