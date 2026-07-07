# Evidence: /task-to-pr skill verification

Branch `add-task-to-pr-skill` adds `.claude/skills/task-to-pr/SKILL.md`.
This transcript shows the skill is discoverable as a user-invocable skill and that every concrete command/path it documents is accurate against the repo.

## 1. Skill registers in a live Claude Code session

The skill appears in the session's available-skills list, loaded from the frontmatter of the committed file:

> task-to-pr: The standard workflow for taking one GitHub issue from start to a merged PR in this repo, using isolated worktrees and the no-mistakes gate. Use when starting work on an issue, asking how to ship or gate a change, or when unsure of the branch/worktree/PR flow.

This is the end-user surface: typing `/task-to-pr` in Claude Code resolves to this skill.

## 2. Frontmatter parses and matches the directory name

```text
$ python3.9 -c "import yaml; meta = yaml.safe_load(open('.claude/skills/task-to-pr/SKILL.md').read().split('---')[1]); print(meta)"
{'name': 'task-to-pr',
 'description': 'The standard workflow for taking one GitHub issue from start to a merged PR in this repo, ...',
 'user-invocable': True}
```

## 3. The pytest node ID cited in the skill exists and passes

Commit 55354ff fixed this node ID; verified against the current tree:

```text
$ pytest "tests/test_frequency.py::TestFrequencyRunEndToEnd::test_run_completes_without_error" -v
tests/test_frequency.py::TestFrequencyRunEndToEnd::test_run_completes_without_error PASSED
======================== 1 passed, 18 warnings in 3.14s ========================
```

## 4. Gate-specifics section matches `.no-mistakes.yaml`

Skill claims vs actual gate config:

| Skill says | `.no-mistakes.yaml` has |
|---|---|
| Test step: `pytest tests/ -v` then `python -m scripts.gate_metrics_smoke` | `test: "pytest tests/ -v && { python -m scripts.gate_metrics_smoke || python3.9 -m scripts.gate_metrics_smoke; }"` |
| Lint step: deterministic no-op (`true`) | `lint: "true"` |
| Plots saved under `.no-mistakes/evidence/` | `test.evidence.dir: .no-mistakes/evidence`, `store_in_repo: true` |

## 5. No duplicate WORKFLOW.md at the repo root

The user asked for the workflow to live only as the skill; the root file is gone:

```text
$ ls WORKFLOW.md
ls: WORKFLOW.md: No such file or directory
$ git ls-tree 55354ff --name-only | grep -i workflow
(no output)
```

## 6. Worktree path is generalized

Step 2 of the skill uses the placeholder form `git worktree add -b <n>-<slug> <worktree-root>/<n>-<slug> main` rather than a machine-specific absolute path (the fix in commit 55354ff).
