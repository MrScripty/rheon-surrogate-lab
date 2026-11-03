# Learned momentum readout and accumulated pressure controls

The negative spatial transpose of the pretrained neural flux convolution matches
the independent native momentum primitive on 144 pressure cases; maximum relative difference is 2.69384e-08.
This derives coefficients from learned weights, with zero new optimizer updates.
The native primitive is an offline teacher only and performs no pressure solve.

Cold controls use true future pressure on 16 consumed training scenes. They omit
nonlinear transport and viscosity. These are offline ceilings, not causal predictions.

| Future steps | Worst velocity error using pressure sum | Worst using last pressure times horizon |
| --- | ---: | ---: |
| 1 | 1.13779e-05 | 1.13779e-05 |
| 2 | 4.95703e-05 | 0.0969334 |
| 3 | 0.000160926 | 0.212752 |
| 4 | 0.00033819 | 0.331024 |

The two-, three- and four-step last-pressure controls miss0.004 despite exact
future endpoint pressure. Pressure sums pass these cold velocity controls, but
supply no pressure forecast, tracer forecast, warm accuracy or own-history claim.
The existing decoder source/weights remain unchanged. A future candidate must
produce all pressures causally and count every learned readout in its block cost.

[Complete native qualification and controls](../../../runs/full-ai-next-priorities-v1/tied-momentum-qualification-01.json) · [Registration](tied-momentum-qualification-registration-01.json)
