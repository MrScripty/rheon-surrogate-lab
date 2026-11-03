# Paired pressure accuracy on GPU

Fresh CUDA accuracy on the same 128 consumed native requests per model.
All input/reference hashes pair exactly. Weights and captured architectures stayed unchanged.
No fresh timing or independent-validation claim. These are pressure-only proposals.

| Model | Worst relative equation residual | Worst mean-free pressure relative L2 |
| --- | --- | --- |
| C11 | 0.28487942 | 0.2677137 |
| C15 | 0.29309019 | 0.20646776 |
| C16 | 0.29100167 | 0.26086888 |
| C17 | 0.2899827 | 0.13604492 |
| C19 | 0.27079658 | 0.14836919 |
| C20 | 0.29136274 | 0.18021903 |
| C24 | 0.10644803 | 0.10565335 |
| C25 | 0.10745332 | 0.15049089 |
| C26 | 0.094272124 | 0.15028939 |
| C36 | 0.066551391 | 0.08113226 |
| C37 | 0.048917066 | 0.053822136 |
| C44 | 0.0039826222 | 0.013450717 |
| C46 | 0.0040508705 | 0.013275492 |
| C47 | 0.0029634214 | 0.0075183027 |
| C48 | 0.0043981399 | 0.009792512 |
| C49 | 0.0036985233 | 0.009369155 |
| C50 | 0.0040036748 | 0.0063411958 |
| C51 | 0.0029722563 | 0.0076667189 |
| C52 | 0.0040257276 | 0.0068653203 |
| C53 | 0.0039001205 | 0.0060215373 |
| C55 | 0.0040787795 | 0.0086993349 |
| C106 | 0.0067875667 | 0.015504815 |
| C107 | 0.0010294371 | 0.0081532901 |

C53 has lower worst pressure-field error than C107 on this population, while
C107 has lower worst equation residual and lower pressure error on 122 of 128
individual requests. Neither metric establishes superiority
on all cases, larger domains or complete trajectories. The retained result
includes every per-case output, denominator and field hash, with separate
density and grid summaries in the [comparison report](reports/pressure-GPU-portfolio-01.json).

The full applicable portfolio benchmark and native project migration remain open.
