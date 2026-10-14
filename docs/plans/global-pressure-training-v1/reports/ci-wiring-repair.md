# Required CI custody gate repair

Parent independent review found that the original green model-contracts job
115836985954 at 9ac0998 ran 45 tests with four RestoreCustodyContract skips. The
workflow checked out exact accepted Tuldok but did not pass its owner source path,
so those tests looked for a task-local /tmp source absent on GitHub. Green status
therefore did not establish complete custody-contract CI coverage. Local tests
had the source available and ran without skips; that did not repair GitHub wiring.

The required test step now explicitly sets TULDOK_CUSTODY_SOURCE to
`${{ github.workspace }}/generic-contracts/recipe_process.py`. Its unchanged
checkout is pinned to f7543fb78f6f1cb52e45dd315efa17a4b2fe6fa6. The same discovered
suite now fails on test failures, any skips or zero tests. No test, trainer, model,
dataset, checkpoint, Run receipt or negative diagnostic was changed.

The exact gate code extracted from the YAML ran all45 tests with zero skips and
exit0 using accepted owner SHA2569bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999.
The missing-source negative ran45 tests with4 skips and exit1, demonstrating the
required failure. See [local qualification](ci-wiring-repair.json). Independent
scoped review and repaired exact-head CI are required before parent acceptance;
the draft PR records their final outcomes and links. No new scientific training,
final arrays or merge was performed. Existing historical CNN CI remains separate.


Independent GPT-6.1 Sol High scoped review is CLEAR. The reviewer separately
extracted and executed this same YAML gate, reproducing45/0/exit0 with the pinned
source and45/4/exit1 with a missing source. The exact checkout/path match and the
gate rejects failures/zero discovery by construction. Source/test/native/original
scientific evidence byte preservation was confirmed. Repaired GitHub zero-skip
coverage remains a required exact-head publication check.
