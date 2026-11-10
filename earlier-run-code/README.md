# Earlier run code

The code and settings files that the lab's earlier training runs executed, in the versions they ran.
Those runs worked from staged copies under `runs/`, which Git ignores, and these versions were never
committed anywhere else. They are kept here so the earlier recipes in `projects/surrogate-lab/recipes/`
name files that are in Git.

A file from `runs/<path>` is here at `<path>`. Each distinct file is here once, at the first of its
paths in sorted order; every recipe that ran an identical file points at that copy. Archives, arrays,
weights and record files (checksums, witnesses, receipts, qualification records) are not copied.

`scripts/build_tuldok_project.py` writes this folder and removes anything it no longer needs. Do not
edit it by hand.
