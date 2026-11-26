"""Identify training code by its published Git commit."""
import os
from pathlib import Path
import re
import subprocess

BRANCH = "codex/weaver-original95-runnable"
REF = "refs/heads/" + BRANCH
URLS = {"git@github.com:distanceLu/AcceRL.git", "https://github.com/distanceLu/AcceRL.git"}

def git(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True,
                            env=dict(os.environ, GIT_TERMINAL_PROMPT="0"), timeout=120)
    if result.returncode:
        raise ValueError(f"git {args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def revision(repo, expected=None):
    if Path(git(repo, "rev-parse", "--show-toplevel")).resolve() != repo.resolve():
        raise ValueError("Expected the training repository root")
    commit = git(repo, "rev-parse", "HEAD")
    if expected is not None and commit != expected:
        raise ValueError(f"Expected Git commit {expected}, found {commit}")
    if git(repo, "status", "--porcelain=v1", "--untracked-files=normal"):
        raise ValueError("Commit the intended changes before starting a new training run.")
    return commit


def published(repo, push=False):
    commit = revision(repo)
    if git(repo, "branch", "--show-current") != BRANCH:
        raise ValueError("Training starts from the " + BRANCH + " branch")
    if git(repo, "remote", "get-url", "origin").removesuffix("/") not in URLS:
        raise ValueError("Expected origin to point to distanceLu/AcceRL")
    if push:
        if git(repo, "remote", "get-url", "--push", "--all", "origin") not in URLS:
            raise ValueError("Unexpected Git push destination")
        git(repo, "push", "origin", commit + ":" + REF)
    remote = git(repo, "ls-remote", "--exit-code", "--heads", "origin", REF).split()
    if remote != [commit, REF]:
        raise ValueError("The local revision differs from the published maintenance branch.")
    return revision(repo, commit)


def source_revision(repo, branch_head, requested):
    """Resolve an explicit historical run against the verified maintenance branch."""
    if requested is None:
        return branch_head
    if not re.fullmatch(r"[0-9a-f]{40}", requested) or git(repo, "cat-file", "-t", requested) != "commit":
        raise ValueError("A historical training run requires a full commit SHA")
    git(repo, "merge-base", "--is-ancestor", requested, branch_head)
    return requested


def pin_worktree(repo, runtime, commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected a full Git commit ID")
    if runtime.exists():
        common = ("rev-parse", "--path-format=absolute", "--git-common-dir")
        if git(repo, *common) != git(runtime, *common) or git(runtime, "branch", "--show-current"):
            raise ValueError("Existing runtime is not this repository's detached worktree")
        revision(runtime)
        git(runtime, "checkout", "--detach", commit)
    else:
        git(repo, "worktree", "add", "--detach", str(runtime), commit)
    return revision(runtime, commit)
