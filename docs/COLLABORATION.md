# Working together

Shared `main` owns the agreed baseline, unchanged author snapshots, reproducible source, configs and protocol. Each person develops in a separate branch. On this Mac, `main/` and `Enze/` are two worktrees sharing one Git history; edits in Enze do not alter main files.

Only main is uploaded initially. The teammate's GitHub username has not been supplied and no invitation is sent yet. No personal Enze branch, class archives, credentials, large tick data or vendor data is uploaded.

Typical personal workflow:

```bash
cd Project/Enze
git status
git add src scripts tests docs configs   # explicit paths, inspect staged diff
git diff --cached
git commit -m 'Describe a tested research change'
git fetch origin
git merge origin/main                  # discuss conflicts; do not overwrite work
```

After discussing which personal commits are shared, merge or cherry-pick those commits into main and push `git push origin main`. Do not use `git push --all`, `--mirror`, or blindly stage ignored data. The Enze worktree has push disabled with a local hook for now. Its branch-specific push URL is invalid as an additional guard.

For a teammate, clone the private repository after invitation, create a personal named branch, then open a PR when a change is ready. Each person must acquire their own data entitlement and rebuild locally. Agree data and evaluation protocol before comparing model results. Shared upstream Git mirrors are stored locally at `Project/upstream_history/`; vendor source files in main are complete snapshots, not truncated examples.

Do not duplicate the professor's futspread source in weekly assignment folders. Its original course-root source and provenance remain unchanged. If reusing its API, add the course root to sys.path and pass the current project's own local data root explicitly, per the course AGENTS.md.
