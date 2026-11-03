# Integration findings

Medium, deferred: new selected-frame/conditioned models require a versioned
comparison deployment adapter. The existing exporter reads
`reservation.configuration.snapshot_steps`, while selected views use
`reservation.sources`; its bridge also does not invoke the conditioning model's
explicit configuration selection. It therefore does not support these experiment
snapshots. Distinct architecture profiles identify that incompatibility but are
not by themselves an automatic adapter migration. The original deployed
sealed-water model and published Rheon integration remain unchanged.

Owner: lab deployment/comparison integration. Next disposition: add explicit
supported-profile admission, conditioning/history reconstruction and fresh CPU
startup/prediction/lifecycle evidence when selecting a new comparison candidate.
Do not overlap the separately maintained configurable Rheon solver architecture
or reinterpret same-shaped tensors through the old model. Current model accuracy
remains unqualified; no new comparison deployment was performed by this tranche.
