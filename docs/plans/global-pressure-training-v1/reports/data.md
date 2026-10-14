# Actual flat development input qualification

The final packet binds native build6 and the immutable slice1 capture. It has
288 train rows (12 complete trajectories) and96 validation rows (4),24 frames per
history. Original RHS, previous/reference pressure and qualified native full-cell
residual remain float64; physical conditioning and explicit row mappings are
numeric and typed. No final/full reservation appears. See [compact receipt](data.json).

Actual flat files: two train shards18,890,920bytes each; validation12,594,856;
origin1,150,372; splits70,478. Aggregate51,597,546bytes. Each is below32MiB;
splits below256KiB; aggregate below256MiB. No compressed-size admission. Metadata
and original-field/raw/native-derived hashes are explicit. External residual
attestation anchor:
`26c33fac5df17b66fadba036368f2008ec172a860231ff75822cd2ef9dba11b8`.

The packager freshly verified all384 original frames before reading original
bytes; verification took19s approximately (exact receipt). Packing/native residual
calls/roundtrip took3.5467s afterward. Independent numerical and standards readers
rechecked actual whole-history lineage, original per-row raw hashes, all three
shards and the frozen external residual origin digest. Native/Torch full residual
comparison across384 rows differs at most4.336808689942018e-19, with exactly zero
gauge-cell discrepancy in the independent run. Every original full-history
fingerprint is unique; shared initial previous-pressure zeros remain diagnostics.
The standalone reader validates packed bytes/lineage and does not rerun native
physics or read opaque old source paths. All19 data admission tests pass.

Packet1 remains preserved as dated support. Native input arithmetic is unchanged
and all three numeric NPZ files are bitwise identical across packets1/2; packet2
updates provenance for the fair full-budget temporal baseline source. Final
project uses packet2. No failed case was dropped or tolerance changed.
