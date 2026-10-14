# Actual owner-local CPU runtime

Exact generic Tuldok sourcef7543fb78f6f1cb52e45dd315efa17a4b2fe6fa6 is an isolated
clean reference. Original Tuldok checkout and upstream code remain unchanged.
Task-private PostgreSQL17.11 uses workspace-extracted authenticated Debian
packages and a private local socket; no system service/global package mutation.
The actual schema8catalog is619034b1109f45faac099835ee9a9065.

Actual captured snapshot
`6367d0fa7eb989d53cc49d6ed47f937a4f63e68655de9acaf84c7f7d94838c77`
contains21,073files/870,900,515bytes, with3,879,376canonicalmanifestbytes. CPU
Torch2.8.0+cpu/NumPy2.3.5 and exact package versions are retained with licenses and
metadata. Optional unsupported Torch tool/test ELFs were explicitly excluded;
mandatory shared-memory manager remains. Initial inventory refusal is preserved.

Existing production runtime qualification completed actual CPU forward/backward,
NumPy raw identity and safe tensor reload, cancellation with verified drain/no
artifacts, and catalog reopen retaining exact completed artifacts. These neutral
Runs are distinct from scientific training. Original source runtime was not
modified or removed; no filesystem-absence claim. See [compact proof](runtime.json).

Scope is actual owner-local captured execution, not hosted/cloud/GPU qualification.
Tuldok provides reviewed cooperative same-UID custody, not hostile-code or network
isolation.600s child wall belongs to existing custody.4GiB RAM is a process
high-water acceptance/stop budget checked by the reviewed trainer before updates,
every20updates and before publication; it is not a generic OS aggregate limit.
Source code uses one Torch/inter-op thread and no multiprocessing/helper calls.
Actual scientific project/preflight/grant and train/restore evidence follows.
