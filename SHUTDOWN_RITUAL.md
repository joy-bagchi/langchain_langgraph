Daily GitHub Remote Sync Check

Before I close for the day, inspect this repo and confirm that my local working tree and GitHub remote are fully synced.

Please check:

1. Current branch
2. git status
3. Uncommitted files
4. Untracked files
5. Local commits not pushed to origin
6. Remote commits not pulled locally
7. Whether origin points to the correct GitHub repo
8. Whether the latest important project files exist on remote GitHub

Specifically verify that the following project area is present remotely:

agentic_vol_regime_app

If anything is not synced, explain exactly what is missing and propose the safest fix before taking action.

Do not delete or overwrite local work.
Do not force push unless I explicitly approve.
Prefer safe commands:
- git status
- git branch -vv
- git remote -v
- git fetch origin
- git log --oneline --decorate --graph --all -20
- git diff
- git diff --staged
- git push origin <current_branch> if local commits are ahead and safe to push

Final output should be:

Remote Sync Status: PASS or FAIL

If PASS:
Confirm local branch, remote branch, latest commit hash, and that GitHub remote contains the latest project files.

If FAIL:
List exactly what is unsynced and the recommended fix.