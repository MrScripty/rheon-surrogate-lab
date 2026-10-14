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
Actual scientific project `0b4606a84eec4c698f9649a97fe59e20` was assembled
from clean committed a9345a4 sources. Training capture was 55,623,792 bytes,
including 51,597,546 input bytes; restore capture was 58,270,164 bytes.
Both independent reviewers cleared each exact preflight before its separate owner
grant. The same captured snapshot executed the actual 200-update training and
zero-update fresh restore. [Training](training.md), [restore](restore.md), and
[scientific catalog reopen](catalog-reopen.json) report exact identities. The
outer owner host interpreter was Python 3.13; the actual recipe target was the
captured Python 3.12.14, proven by its runtime observations. These are distinct.

The actual current-catalog Runs UI was observed after service reopen in Chromium
at 1280×900 and 390×844. It rendered the completed training progress, hash-verified
summary, custody and artifact inventory; checkpoint download matched actual trained
weight SHA256 at both widths, with no page errors or document horizontal overflow.
An initial bearer-header-only browser download canceled while page rendering
worked; using the same existing scoped agent credential as an HttpOnly cookie
made ordinary navigation/download work. This test qualifies neither interactive
login nor UI training submission. See [observations](ui.json); screenshots remain
in ignored task-local evidence storage.
