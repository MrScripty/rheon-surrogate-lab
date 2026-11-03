# Float32 target calibration

The current learned-output profile is relative full original-RHS pressure
residual <= 1e-6, with the original absolute residual floor 1e-12. This is a
precision decision, not a float32 range limit. The seven-channel model remains
float32 with its exact existing weights and continuous learning loss. The strict
1e-8 native reference/corrector and every historical result remain unchanged.

All physical limits remain enforced: predicted divergence <= 1e-7 s⁻¹; actual
divergence after f32 velocity correction <= 1e-5 s⁻¹; original net-flux limits,
zero wall-normal velocities, pressure gauge/finiteness and Courant <= 0.5.
The versioned profile produces an additional model acceptance assessment; it
does not overwrite native outcomes or claim altered solver iteration counts.

Calibration reproduced the actual output path on reference-minus-base pressure
corrections: divide by train-only pressure scale, round to float32, rescale,
fix the correction gauge and add previous pressure. Fresh native diagnostics
cover all 96 train frames before any of the 48 validation frames. Maximum train
relative residual is 4.7665255695e-7. A factor-two margin rounded up selects
1e-6, recorded in selection.json before validation. Validation maximum is
3.5748981451e-7; no threshold tuning used validation or learned outcomes.

All 144 rounded ideal corrections pass the new profile; none pass the original
strict native residual gate. Maximum predicted divergence is 3.5097870018e-8
s⁻¹, and maximum actual rounded-velocity divergence is 1.7881393433e-7 s⁻¹,
both within their unchanged limits. These are ideal quantization diagnostics,
not evidence that the learned network solved these frames.

The twelve actual restored seven-channel predictions were also freshly checked:
0/12 pass the new model-output target, with worst relative residual 0.6177574216.
The precision ceiling is no longer the acceptance obstacle at 1e-6; learning
error remains substantial. No training, optimizer updates, weight changes,
autonomous rollout, GPU work or acceleration claim occurred.

Five focused policy checks cover calibration margin, unchanged native data,
physical/invalid-output rejection, zero-RHS absolute floor and malformed profile
limits. The existing native-response validator and independent full operator
check every one of the 156 fresh diagnostics. The driver validates selected
native/source/packet identity, train-only normalization, completed managed restore
identity and exact shared artifact bytes. Evidence lives in
runs/float32-pressure-target-v1-1, including per-case native records, frozen
profile, selection and measured report. acceptance.json captures the report.

The new current profile and external diagnostic report are exposed through the
working Tuldok project. Actual calibration and model fields were published to
Rheon with explicit provenance labels. Original managed Runs remain the owners
of training/restored checkpoints; this check does not fabricate another Run.
The private feature branch is retained for continued review; no push or merge.
