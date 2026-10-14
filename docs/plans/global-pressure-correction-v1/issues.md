# Issues and dispositions

| ID | Finding / evidence | Owner | Disposition |
| --- | --- | --- | --- |
| I1 | CNN loss/restore success coexists with native0/16; inspected sources and receipts | Lab evidence owner | Preserved unchanged; never qualified acceleration |
| I2 | Earlier fixed-box global model did not establish advantage over analytic cosine; old reservation consumed | Lab model owner | Analytic mandatory; new geometry/trajectory/regime reservation |
| I3 | Existing sealed warm-start patch is based on older34235d3, not trajectory2209012b | Next native implementation owner | Pending prerequisite: isolated forward-port/source/API proof before model code; stop if unavailable |
| I4 | Captured CPU profile excludes CUDA; full raw dataset exceeds captured-input limits | Lab/Tuldok boundary owners | CPU smoke within existing bounds; separate local-GPU full path and external-evidence scope |
| I5 | No applicable lab/workspace AGENTS.md or .agents/skills present; standards source referenced by lab was resolved | Task agent | Read existing standards188beda and record exact scope; no nonexistent instruction inferred |
| I6 | Geometry/regime grouping, teacher-state train vs own-state evaluation, f32 divergence and transport drift need explicit qualification | Next scientific implementation owner | Protocol and independent oracles prescribed; all E criteria pending |
| I7 | Source timings may be worse despite fewer iterations; inference/reset fallback/setup costs matter | Lab performance owner | Seven matched-quality methods, complete wall intervals and negative outcome publication required |

| R1 / P1 | Review: full raw data21.8GiB exceeds12GiB cap | Design author | Repaired to32GiB with measured transient/free-space admission; independent recheck closed |
| R2 / P2 | Review:4hour cap excludes unspecified cold workload | Design author | Three resident+three cold repeats explicit,705600full method-frames,16hour conditional cap and train-only projection; independent recheck closed |
| R3 / P2 | Review: full E3/E4/E6 improperly inherit managedCPU200-update scope | Design author | Full10k localGPU/restore parity/bounded managedCPU+external evidence mapping; independent recheck closed |
