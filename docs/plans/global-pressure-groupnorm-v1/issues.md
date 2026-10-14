# Issues

Pretraining acceptance remains in progress. GroupNorm remains a hypothesis; trained behavior and solver usefulness are unknown.

Independent preliminary review found wrong project-root calculation, dangling captured-source fallback and permissive qualification metric inventory. Owners corrected each and added focused checks; final source review remains required.

Combined discovery exposed a separate ambient-import obstruction: the byte-pinned historical test fixture inserts its source directory into sys.path, causing a later plain `import project` to select the historical six-output assembler. The original test bytes remain unchanged. The new test-only loader restores its prior path, and the new assembler/declaration tests use explicit owned module paths. The first combined28-test failure is preserved in ignored logs; final combined tests and independent review must prove repair.

The existing API credential expired on a read-only inspection (401). A new inspect/edit-only owner-local credential is used for preflight; this is not a training grant. The first noncanonical runtime descriptor was rejected before target execution and retained; the corrected production-bootstrap observation passed.
