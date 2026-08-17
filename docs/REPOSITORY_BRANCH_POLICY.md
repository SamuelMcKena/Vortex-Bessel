# Repository Branch Policy

**Established:** 2026-08-14

This document records the repository cleanup performed after the GitHub default branch was found to point at an effectively empty historical freeze tree.

## Authoritative branch

`main` is the authoritative current project line.

At the time this policy was established, `main` contains the complete validated Phase 2E -> Phase 2I lineage plus the subsequent Phase 2I presentation-figure audit/selection work.

All new development should branch from `main` unless a deliberately isolated historical reproduction is required.

## GitHub default-branch compatibility

GitHub still names `vortex-report-freeze-v1` as the repository default branch because the current connector cannot change the repository-level default-branch setting.

Until the GitHub setting is manually changed to `main`, `vortex-report-freeze-v1` is intentionally kept fast-forwarded to the same current commit as `main` so tools that implicitly resolve the default branch do not see the old empty tree.

After the repository default is changed to `main` in GitHub Settings -> Branches, `vortex-report-freeze-v1` should be treated as a compatibility/historical branch only.

The original broken/empty default state is preserved at:

- `broken-default-backup-20260814`

Nothing from that state was deleted.

## Validated preservation branches

The following immutable-style archive branches pin important validated milestones:

- `archive/validated-phase2e-axicon-v3-20260811`
- `archive/validated-phase2f-propagation-audit-20260811`
- `archive/validated-phase2g-digital-twin-20260811`
- `archive/validated-phase2h-vector-axicon-20260812`
- `archive/validated-phase2i-experimental-closure-20260812`
- `archive/presentation-figure-audit-20260814`
- `archive/legacy-conference-workshop-figures-20260811`

These branches preserve exact historical heads for reproducibility and should not be used as normal development branches.

## Pull-request cleanup

PRs #1-#11 were closed during the cleanup after their work was either:

1. explicitly temporary/diagnostic/remote-run only;
2. superseded by a later validated stacked phase; or
3. already present in the ancestry of current `main`.

Closing those PRs did **not** delete their branches, commits, CI records, artifacts, comments, or scientific provenance.

The original numerical/calibration/report-claim boundaries documented in those PRs remain in force. Repository cleanup must never be interpreted as upgrading a synthetic or calibration-limited result into an experimental claim.

## Development rule going forward

For new work:

1. start from current `main`;
2. use a short descriptive feature branch;
3. keep temporary remote-run/diagnostic branches explicitly labelled as temporary;
4. open one PR for real reviewable work rather than a long stack of unresolved drafts where practical;
5. validate physics/numerics with the relevant CI gates;
6. promote completed work back onto `main`;
7. for major validated milestones, pin the exact validated head under `archive/validated-*`;
8. do not move an archive branch after it has been created.

Presentation-specific generated figures may live on `main` when they are derived from and documented against validated numerical evidence, but they must not silently replace or relabel the underlying scientific evidence.

## Current practical rule for tools and AI coding sessions

When opening this repository in Codex, ChatGPT, GitHub tools, Codespaces, or a local clone, use `main` explicitly until the GitHub default branch has been switched to `main` in repository settings.
