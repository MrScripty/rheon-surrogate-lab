# Named pressure checkpoints: common GPU accuracy

All 127 available named checkpoints completed the same 128 consumed native requests.
Every saved pressure field was rehashed and independently rescored. C01/C02 remain unavailable.
No new fits, CPU neural forwards, independent-validation or fresh latency claims.

| Checkpoint | Name | Worst common equation relative L2 | Worst mean-free pressure relative L2 |
| --- | --- | --- | --- |
| C03 | Variable grid · 10 channels | 0.47514674 | 0.29085258 |
| C04 | Variable grid · 7 channels | 0.47508074 | 0.29089128 |
| C05 | CUDA portability · 20 updates | 0.47509411 | 0.29082936 |
| C06 | Original 256-simulation water cohort | 0.19846746 | 0.123598 |
| C07 | fit-one | 0.27569873 | 0.23065381 |
| C08 | long-256 | 0.19760061 | 0.1191549 |
| C09 | scale-256 | 0.18913792 | 0.12126727 |
| C10 | scale-1024 | 0.20792456 | 0.13061603 |
| C11 | constant-low | 0.28487942 | 0.2677137 |
| C12 | constant-high | 0.27333003 | 0.20990593 |
| C13 | delayed-decay | 0.27453207 | 0.20531131 |
| C14 | scalar-units | 0.44035031 | 1.6662778 |
| C15 | no-groupnorm | 0.29309019 | 0.20646776 |
| C16 | residual-nonlinear | 0.29100167 | 0.26086888 |
| C17 | residual-linear | 0.2899827 | 0.13604492 |
| C18 | residual-recurrent | 0.22180619 | 0.16734697 |
| C19 | residual-physical-loss | 0.27079658 | 0.14836919 |
| C20 | residual-linear-5000 | 0.29136274 | 0.18021903 |
| C21 | stability-modes | 0.032820812 | 0.016844522 |
| C22 | stability-scale | 0.14641367 | 0.079388778 |
| C23 | stability-combined | 0.006128791 | 0.0027531737 |
| C24 | stability-modes-single-pass | 0.10644803 | 0.10565335 |
| C25 | stability-combined-single-pass-lr1e4 | 0.10745332 | 0.15049089 |
| C26 | multigrid-modes-single-pass | 0.094272124 | 0.15028939 |
| C27 | multigrid-modes-two-pass | 0.047725731 | 0.04227554 |
| C28 | multigrid-cohort16-2pass | 0.019291235 | 0.016669688 |
| C29 | cohort-two-seed2 | 0.024295939 | 0.018699412 |
| C30 | cohort-two-seed3 | 0.019836452 | 0.025842432 |
| C31 | cohort-one-control | 0.073253374 | 0.093818012 |
| C32 | cohort-four-control | 0.006450019 | 0.0055660151 |
| C33 | cohort-two-shape-feedback | 0.017396274 | 0.019278994 |
| C34 | shape-two-seed2 | 0.015272181 | 0.01346159 |
| C35 | shape-two-seed3 | 0.020026485 | 0.018309088 |
| C36 | shape-one-control | 0.066551391 | 0.08113226 |
| C37 | shape-one-longer | 0.048917066 | 0.053822136 |
| C38 | two-pass-10000 | 0.0096752633 | 0.0098655587 |
| C39 | coverage-only | 0.032289648 | 0.022446335 |
| C40 | feedback-only | 0.0083245411 | 0.014911085 |
| C41 | supplement-control | 0.011665529 | 0.011980936 |
| C42 | targeted-feedback | 0.010799692 | 0.010405588 |
| C43 | extra-early | 0.004725959 | 0.0078238244 |
| C44 | extra-late | 0.0039826222 | 0.013450717 |
| C45 | control | 0.0056079029 | 0.0097331973 |
| C46 | first-pressure | 0.0040508705 | 0.013275492 |
| C47 | gentle-010 | 0.0029634214 | 0.0075183027 |
| C48 | gentle-025 | 0.0043981399 | 0.009792512 |
| C49 | lowband | 0.0036985233 | 0.009369155 |
| C50 | weight001 | 0.0040036748 | 0.0063411958 |
| C51 | weight003 | 0.0029722563 | 0.0076667189 |
| C52 | current-early | 0.0040257276 | 0.0068653203 |
| C53 | current-late | 0.0039001205 | 0.0060215373 |
| C54 | current003 | 0.0045794348 | 0.0067340122 |
| C55 | current001 | 0.0040787795 | 0.0086993349 |
| C56 | full-coarse | 0.0098092483 | 0.015527807 |
| C57 | two-pass-control | 0.0039117632 | 0.0068503674 |
| C58 | four-pass-physics | 0.0036007486 | 0.0052917723 |
| C59 | four-pass-teacher | 0.0035635852 | 0.0053016238 |
| C60 | balanced-control | 0.004393752 | 0.0066126765 |
| C61 | low-mode-tail | 0.0044076738 | 0.006559743 |
| C62 | single-mode-fit | 0.0078423797 | 0.019326517 |
| C63 | frequency-curriculum | 0.0041662696 | 0.0057310315 |
| C64 | combined-fundamentals | 0.0068547551 | 0.015976136 |
| C65 | axis-isolation | 0.0065443352 | 0.017042724 |
| C66 | shape-isolation | 0.005882819 | 0.017965911 |
| C67 | shape-later-decay | 0.0064857626 | 0.0171098 |
| C69 | balanced-shapes | 0.0047359065 | 0.0099685227 |
| C70 | c53-a | 0.0054914175 | 0.012387427 |
| C71 | c53-d | 0.0052965075 | 0.012479613 |
| C72 | c44-u | 0.0067723742 | 0.022052821 |
| C73 | c44-a | 0.0070223308 | 0.021384582 |
| C74 | c44-d | 0.0064598383 | 0.021523997 |
| C75 | c37-u | 0.0072754935 | 0.011298825 |
| C76 | c37-a | 0.007451432 | 0.011092771 |
| C77 | c37-d | 0.0072899657 | 0.01118007 |
| C78 | domain-small-control | 0.0049530981 | 0.0068566247 |
| C79 | domain-expanded | 0.006886264 | 0.00855929 |
| C80 | teacher-half | 0.0047538663 | 0.0063801845 |
| C81 | teacher-quarter | 0.0048968677 | 0.0065439187 |
| C82 | spacing-data-control | 0.0049617685 | 0.0068289891 |
| C83 | numerical-coarse | 0.00085887549 | 0.0044392805 |
| C84 | sparse-anchors | 0.015850706 | 0.0028033987 |
| C85 | learned-coarse | 0.0049418269 | 0.0064126848 |
| C86 | spacing-zero-feature | 0.0049618307 | 0.0068284571 |
| C87 | spacing-conditioned | 0.0048460059 | 0.0068194332 |
| C88 | coarse-head-rate | 0.0024461164 | 0.0065669124 |
| C89 | v2-small24 | 0.0062369483 | 0.006942531 |
| C90 | v2-mixed16 | 0.0042363637 | 0.0081601044 |
| C91 | v2-retain24 | 0.0042299585 | 0.0074792796 |
| C92 | v2-headbase | 0.0022252286 | 0.0079501354 |
| C93 | v2-headshape | 0.00088136444 | 0.0031069917 |
| C94 | v2-num4 | 0.00046530894 | 0.0016703933 |
| C95 | v2-num6 | 0.00015193494 | 0.00016009863 |
| C96 | v2-headshape-s25 | 0.00096555504 | 0.0031856159 |
| C97 | v2-headshape-s26 | 0.0010028307 | 0.0037142945 |
| C98 | v2-num6-s25 | 0.00015173337 | 0.00016779267 |
| C99 | v2-num6-s26 | 0.00014749795 | 0.00016969859 |
| C100 | v3-replay | 0.00092314577 | 0.0030819237 |
| C101 | v3-native | 0.00085317878 | 0.0052973367 |
| C102 | v3-c37half | 0.071035387 | 0.013365002 |
| C103 | v3-c44half | 0.061543352 | 0.012007447 |
| C104 | v4-student75-distill | 0.014174098 | 0.027748437 |
| C105 | v4-student75-control | 0.00085465419 | 0.0063920437 |
| C106 | v4-student50-distill | 0.0067875667 | 0.015504815 |
| C107 | v4-student50-control | 0.0010294371 | 0.0081532901 |
| C108 | v4-replay | 0.000811671 | 0.0051720163 |
| C109 | v4-gcr | 0.00051156347 | 0.0038993858 |
| C110 | v4-diverse | 0.00079813342 | 0.005205571 |
| C111 | v4-onpolicy | 0.0008633544 | 0.0052733128 |
| C112 | v5-raw | 0.092286375 | 0.22061745 |
| C113 | v5-multilevel | 0.10438272 | 0.23135373 |
| C114 | v5-spectrum | 0.10394946 | 0.23105798 |
| C115 | v6-replay | 0.10461522 | 0.232013 |
| C116 | v6-stencil | 0.17885221 | 0.26139671 |
| C117 | v6-separable | 0.18183158 | 0.26165754 |
| C118 | v6-unrolled | 0.10381402 | 0.23154688 |
| C119 | a-control | 0.0011419362 | 0.0075180436 |
| C120 | a-head | 0.0011061112 | 0.0077217366 |
| C121 | a-tail | 0.00098943151 | 0.0078023168 |
| C122 | a-combined | 0.00098742709 | 0.0079692483 |
| C123 | a-redistributed | 0.0020976896 | 0.015827308 |
| C124 | a-one-stage | 0.01287619 | 0.048103248 |
| C125 | a-head-calibration | 0.00046812096 | 0.0024197173 |
| C126 | a-head-calibration-s2 | 0.00044639379 | 0.0023671402 |
| C127 | a-head-calibration-s3 | 0.00047027596 | 0.0024062009 |
| C128 | a-tail8 | 0.00022551206 | 7.9367856e-05 |
| C129 | a-tail8-s2 | 0.00022551851 | 8.0209849e-05 |
| C130 | a-tail8-s3 | 0.00022552099 | 8.5900931e-05 |

Results are diagnostics. Completing these pressure requests does not establish
full simulation accuracy, own-history stability, generalization or speed.
Density and domain dimensions remain separate in the retained per-case results.
C03/C04 use GPU CNN execution with their captured CPU preprocessing/reconstruction;
the CNN operation bodies match the original sources. Other checkpoints use their
captured CUDA implementations. No trained parameter changed.

[Evidence and density summaries](reports/pressure-common-score-01.json).
The full applicable portfolio benchmark and native project migration remain open.
