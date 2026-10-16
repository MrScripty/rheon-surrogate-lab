# Execution ledger

- Admitted from9fec0f2. Earlier results preserved. The six pre-existing untracked
  graph files/directories are outside the write set and remain user-owned.
- Predeclared pressure targets1e-6,3e-7,1e-7; feedback weights0.1/0.25; original
  seed20261015 and conditional replication seed20261016. Independent8 unused.
- Reuse exact archived harder-data weight0 control, conditioned on matching source,
  initialization and sample streams. No raw simulations or training data added.

- Source qualification: analytical spectral Parseval/amplitude-bin test and existing first-pressure/direction tests passed. Control binding verified11 canonical training files. Configuration captured at3a6ec2d.
- CPU saved-input replay:36 replays on18 identical supplied states,45.978s total. At1e-6/3e-7/1e-7, new control mean calls7.833/8.833/9.278, max11/12/13; old late mean9.222/9.889/11.056, max16/18/20. Both independently checked against the neighbor action. Replay peak individual RSS789880832B. These are local same-RHS errors, not autonomous accuracy.
- Both stricter policies qualified real first-AI-step fixtures in5.417s; full74 consumed scenes admitted. Three new managed projects assembled in3.216/3.266/3.565s (smoke/0.1/0.25). Main project2f65f2a6e21c4710b6070e03e3b7ca9c opened in original cataloga7da4a8de53541a2a0e2aad08005bb71.

- M1 accepted: real CUDA smoke+fresh CPU restore completed and custody drained; managed elapsed592.978s, train recipe6.696s, restore recipe3.217s. Source, CPU replay and real environment claims satisfied. Tuldok runtime setup remains separately measured overhead; no runtime changes in this model experiment.

- Stricter-policy comparison completed:222 episodes (74 native+148 AI), all qualified, no failure. Correctness wall2185.8s; policy analysis0.664s. Neither policy passes all group work/pressure gates. Target3e-7 improves recent8 worst trajectory pressure2.3851e-6→9.5385e-7 against harder-data control, but max calls11→12; original58 worst3.2255e-6 remains above old late3.1448e-6 and mean8.5047 exceeds7.9373. Target1e-7 is not a monotonic autonomous improvement; all rows retained.
- Weight0.1 CUDA5000-update optimizer562.049s, managed train+restore1256.730s, peak CUDA770565632B. Existing input/feedback/initialization match will be bound with the pair after second restore. Weight0.25 now training. Weight0.1 CPU spectral diagnostics overlap that fit after native policy workers drained.

- While weight0.25 fits on GPU, weight0.1 diagnostics completed:18 saved-input replays21.221s and temporal/manufactured amplitude73.136s. At8 calls, worst same-input pressure6.0483e-6 versus archived harder-data control8.3229e-6 on the same18 states. Long9×11×33 pure low1Pa coefficient error2.9783e-5Pa improves control4.6344e-5Pa; high-only leakage3.5035e-4Pa worsens control1.2682e-4Pa. Diagnostic-only; no settings altered.
