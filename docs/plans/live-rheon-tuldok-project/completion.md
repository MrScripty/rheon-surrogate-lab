# Live observation acceptance

The actual lab repo is registered in the existing Tuldok data-postgres catalog
on 127.0.0.1:8091, project 032f4d5b8ab249dc94c4d5f1f8a331c4. Its project view
updates producer activity and selected source text automatically. Current catalog
data and sessions remain intact. A new draft field-sequence record
45bd771a79c446368fe8774bd5f5fe9e contains one whole verified cube trajectory:
24 frames, pressure/tracer/projected u/v/w,16³ cells in 1 m³, spacing 0.0625 m.
Original f32/f64 bytes are preserved. No review or release was admitted.

Rheon's actual egui app is open in external observation mode. Native rendering
was inspected from the desktop, not a browser imitation. It uses physical box
aspect ratios, supports rectangular count arrays, a pressure slice and bounded
retained-frame scrubbing. Captures, native evaluations and normal scoped Tuldok
Run progress have separate observer interfaces. Normal Run/custody remains the
execution authority; external status never fabricates a managed Run.

The real live capture qualified its one-frame train fixture first, then all
384 development frames. The scientific worker/observer composition initially
failed the frozen native caller's single-thread guard before any completed case;
that zero-case failure remains at runs/live-project/native-evaluation-1.
Moving observation to the parent preserved the unchanged native process owner.
The subsequent 32-case teacher-state diagnostic completed: raw 0/32, fallback 32/32,
6474 learned-policy iterations versus5450 baseline. It uses existing restored
predictions and reference tracer; no new model or autonomous rollout is claimed.

Checks: 41existing/new native desktop tests pass; final 3 focused preview tests
include 48×32×24counts with equal 0.03125 m spacing; Clippy warnings are rejected.
Tuldok 21 focused/affected project tests and actual browser activity/source/deep-link
workflows pass. Lab 4 observer tests include the single-threaded scientific-worker
regression. The live cube bundle passes Tuldok's complete parser and real catalog
import. Historical CPU/model/capture/evaluation executable bytes remain unchanged.

Ignored local proof and logs: runs/live-integration; actual publications/captures,
the failed prefix, completed diagnostic and field ZIP: runs/live-project.
The application/source worktrees and live UI processes are deliberately retained
for the user. The private PostgreSQL test service may remain running during this
integration; the user's original application database is distinct. No push/merge
or external dirty-worktree cleanup occurred.

The owner subsequently required cubic cells with 16 cells per metre as the default.
This observation slice does not rewrite historical rectangular-cell data. The
next numerical configuration must derive per-axis counts from shared h = 1/16 m,
qualify that grid independently and reject incompatible model shapes before
training. That numerical change is separate from this UI/observation commit.

Source commits: Tuldok 9eeb653, Rheon a15e160. Their source branches/worktrees are
retained locally; the lab commit binds this observation adapter separately.

Subsequent numerical setup is independently qualified in
[the cubic-cell completion](../cubic-pressure-v1/completion.md). The current
project is 6ed9bfccb7ec4d85ac5849b6c2dc0673; the previous project remains registered.
The user's original desktop simulation and recordings remain intact. A separate
read-only experiment window displays the verified rectangular-count cubic-cell
fields. Training observation was implemented but no additional managed training
Run was exercised for this UI acceptance.
