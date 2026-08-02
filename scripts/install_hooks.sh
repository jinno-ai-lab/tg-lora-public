#!/usr/bin/env bash
# install_hooks.sh -- deploy the checked-in git hooks (scripts/hooks/*) into git's
# hooks directory so the prepare-commit-msg empty-commit guard fires
# automatically at commit time.
#
# WHY A SCRIPT (not an inline Makefile recipe)
#   The deployment used to live inline in the Makefile `install-hooks` target,
#   and that recipe was never tested. tests/test_commit_hook.py installs the hook
#   by hand-rolling its OWN symlink, so it proved "IF installed, the hook fires"
#   but NOT "the hook actually gets installed". A broken recipe (wrong glob, a
#   dangling symlink, or a second hook added but not picked up) would have shipped
#   INERT: a guard that is never installed cannot fire -- exactly the "passes due
#   to its own presence" pathology this iteration's feedback names. Extracting the
#   recipe here lets tests/test_install_hooks.py run the REAL deployment against a
#   sandbox HOOKS_DIR and assert every checked-in hook lands as a RESOLVING
#   symlink whose guard chain is ACTIVE.
#
#   `make install` now depends on `install-hooks`, so the standard repo setup
#   deploys the guard -- a fresh clone / linked worktree (including the make-run
#   path the guard was built to protect) no longer ships inert.
#
# USAGE
#   scripts/install_hooks.sh                           # HOOKS_DIR = git rev-parse --git-path hooks
#   HOOKS_DIR=/tmp/sandbox scripts/install_hooks.sh    # explicit target (tests / custom layout)
#
# ENV
#   HOOKS_DIR   target hooks directory (default: git's hooks path for the CWD)
#   HOOKS_SRC   checked-in hooks source dir (default: <repo>/scripts/hooks)

set -euo pipefail

_self="${BASH_SOURCE[0]:-$0}"
_repo_root=$(cd "$(dirname "$(readlink -f "$_self")")/.." && pwd)
HOOKS_SRC="${HOOKS_SRC:-$_repo_root/scripts/hooks}"

if [ -z "${HOOKS_DIR:-}" ]; then
    if ! HOOKS_DIR=$(git rev-parse --git-path hooks 2>/dev/null); then
        echo "install_hooks: HOOKS_DIR is unset and 'git rev-parse --git-path hooks'" \
             "failed (not inside a git repo?). Set HOOKS_DIR explicitly." >&2
        exit 1
    fi
    # git may return a path relative to the repo root; absolutify so the symlinks
    # stay valid regardless of the CWD git is later invoked from.
    case "$HOOKS_DIR" in
        /*) : ;;
        *)  HOOKS_DIR="$(cd "$HOOKS_DIR" 2>/dev/null && pwd)" ;;
    esac
fi

if [ ! -d "$HOOKS_SRC" ]; then
    echo "install_hooks: hooks source dir not found: $HOOKS_SRC" >&2
    exit 1
fi

mkdir -p "$HOOKS_DIR"
echo "installing hooks into $HOOKS_DIR"
for h in "$HOOKS_SRC"/*; do
    [ -f "$h" ] || continue
    name=$(basename "$h")
    chmod +x "$h"
    src_abs="$(cd "$HOOKS_SRC" && pwd)/$name"
    ln -sf "$src_abs" "$HOOKS_DIR/$name"
    echo "  linked $name -> $src_abs"
done
