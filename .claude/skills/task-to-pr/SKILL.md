---
name: task-to-pr
description: The standard workflow for taking one GitHub issue from start to a merged PR in this repo, using isolated worktrees and the no-mistakes gate. Use when starting work on an issue, asking how to ship or gate a change, or when unsure of the branch/worktree/PR flow.
user-invocable: true
---

# Task → PR Workflow

Concise steps for taking one GitHub issue from start to merged PR in this repo.
The `no-mistakes` gate does all pushing and PR-opening - never `git push origin` or open a PR by hand.

## The loop (one issue = one branch = one PR)

- **1. Pick the issue.** Pull the next open issue(s) by number. Confirm dependencies are merged and the issue's premise still matches the code (e.g. #113 was stale - verify before trusting the issue text).
- **2. Isolate: worktree + branch in one step.** Off a clean `main`:
  ```bash
  git worktree add -b <n>-<slug> <worktree-root>/<n>-<slug> main
  ```
  `-b` creates the branch and the checkout together. Your main checkout stays on `main` and untouched.
  (You can also just create a branch in place - but a worktree keeps `main` clean and lets you run issues in parallel.)
- **3. Implement in the worktree.** Make the change, keep type hints, follow repo conventions (plain `-`, config-driven).
- **4. Test locally until green.** Run the issue's named test first, then the full suite:
  ```bash
  pytest tests/test_frequency.py::TestFrequencyRunEndToEnd::test_run_completes_without_error -v
  pytest tests/ -v
  ```
- **5. Commit on the branch.** Conventional message, `(#<n>)` reference, no agent co-author, no `—`.
- **6. Hand the *branch* to the gate.** Either:
  ```bash
  git push no-mistakes <n>-<slug>     # or run /no-mistakes
  ```
  The gate spins up its **own** disposable worktree, runs review → test → docs → lint, auto-applies safe fixes, escalates judgment calls, then opens the PR to `origin` itself. You pass the branch, not your worktree.
- **7. Drive the gate to green.** Watch and respond as it runs:
  ```bash
  no-mistakes axi status
  no-mistakes axi logs --step test --full
  no-mistakes axi respond --action approve   # approve escalated steps
  ```
- **8. Review the final gated PR.** This is your only manual step.
- **9. Merge, then clean up.** After merge:
  ```bash
  git worktree remove <path>              # remove your worktree
  git push no-mistakes :<n>-<slug>        # delete the remote branch if needed
  ```

## One-time setup (per clone)

- `no-mistakes init` - creates the `no-mistakes` remote and installs the `/no-mistakes` skill. If the remote or skill is missing, run this first.
- Ensure the `git push no-mistakes` allow-rule exists in settings so pushes don't prompt.

## Gate specifics for this repo (from `.no-mistakes.yaml`)

- **Test step:** `pytest tests/ -v` then `python -m scripts.gate_metrics_smoke` (retrains frequency model on the committed dataset, prints the metrics table, saves plots under `.no-mistakes/evidence/`).
- **Lint step:** deterministic no-op (`true`) - no linter is wired. Don't invent one.

## Non-negotiables

- Never `git push origin`; never open a PR by hand - the gate owns both.
- Branch before working; committing issue work directly on `main` is wrong.
- Branch off a clean `main`, not off an existing detached-HEAD worktree, so the gate diffs cleanly against `origin/main`.
- One issue → its own worktree, branch, gate run, and PR.