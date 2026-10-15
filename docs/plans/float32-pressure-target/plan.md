# Float32 pressure-output acceptance target

User authorized keeping learned outputs in float32 and setting an attainable
precision target. Root agent owns this serial slice on the current private
feature branch. Preserve all historical models, losses, gates, results and
managed Runs. No retraining, model architecture change, GPU or publication.

Write set: experiments/float32-pressure-target-v1, this plan and acceptance
evidence, current README/Tuldok pointers. Reuse the exact qualified seven-channel
model, train-only normalization, packet, native binary and response validator.
The new profile changes model-output residual acceptance only. Native reference
and conventional solver thresholds remain their historical strict values, and
all physical divergence/flux/wall/Courant failures remain blocking. No altered
solver iteration count, retrospective pass claim or acceleration claim.

Train-only calibration: reference-minus-base correction divided by saved pressure
scale, converted to float32, rescaled/gauged exactly like the actual head, added
to base. Evaluate every training frame through native physics, use twofold error
margin rounded up to a decimal power, and freeze selection before validation.
The configured candidate is relative residual 1e-6, absolute floor 1e-12. This is
an empirical profile for the current development family, not a universal float32
limit. Check all validation frames and actual restored predictions separately.

Status: complete. Five focused checks pass. Fresh native diagnostics confirm
the train-only 1e-6 selection and acceptance of all 144 float32-rounded ideal
corrections under unchanged physical gates. Actual restored model predictions
remain 0/12. Independent operator checks and old native-response validation
pass on all 156 cases. No model/weight/loss changes or optimizer updates.
Real calibration and learned fields/progress were published through the existing
observers. completion.md and acceptance.json preserve measured results.
