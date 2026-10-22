#!/usr/bin/env bash
# upstream_status.sh — how far NVIDIA/Model-Optimizer has moved, and whether it touched what omk_qad uses.
#
#   bash omk/scripts/upstream_status.sh               # report only
#   bash omk/scripts/upstream_status.sh --sync-main   # also fast-forward the `main` mirror to upstream/main and push it
#
# Branch model (omk/docs/upstream.md): `main` is a pure fast-forward mirror of upstream/main; `omk-main` (default)
# is upstream plus omk's ADDITIVE tree. The watch list is derived from omk_qad's own `modelopt` imports, so it
# follows the code. Exit 1 if omk-main modifies any upstream file (the additive rule is broken).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 2
OURS="omk-main"; OWN_RE='^(omk_qad/|omk/|\.github/README\.md$|\.github/workflows/omk-)'
git fetch -q upstream main || { echo "fetch upstream failed"; exit 2; }
git fetch -q origin || true

BASE=$(git merge-base "$OURS" upstream/main)
echo "== upstream/main $(git log -1 --format='%h %ci %s' upstream/main)"
echo "== omk-main base $(git log -1 --format='%h %ci %s' "$BASE")"
echo "   upstream commits not yet in omk-main: $(git rev-list --count "$BASE"..upstream/main)"

# modelopt modules omk_qad imports -> repo paths (package dir if it is one, else the .py)
mapfile -t WATCH < <(grep -rhoE '(from|import) modelopt(\.[A-Za-z0-9_]+)+' omk_qad --include='*.py' \
  | awk '{print $2}' | sort -u | while read -r m; do p=${m//./\/}; [ -d "$p" ] && echo "$p" || echo "$p.py"; done | sort -u)
echo "== watch list (from omk_qad imports): ${WATCH[*]}"
n=$(git rev-list --count "$BASE"..upstream/main -- "${WATCH[@]}")
echo "   upstream commits touching it: $n"
if [ "$n" -gt 0 ]; then
  git log --format='   %h %ci %s' "$BASE"..upstream/main -- "${WATCH[@]}"
  git diff --stat "$BASE" upstream/main -- "${WATCH[@]}" | sed 's/^/   /'
  echo "   -> merge only after the bit-exactness suite passes (omk/docs/upstream.md)"
fi

bad=$(git diff --name-only "$BASE" "$OURS" | grep -vE "$OWN_RE" || true)
if [ -n "$bad" ]; then echo "== ADDITIVE RULE BROKEN: omk-main modifies upstream files:"; echo "$bad" | sed 's/^/   /'; rc=1
else echo "== additive rule OK: omk-main touches only omk_qad/, omk/, .github/README.md, .github/workflows/omk-*"; rc=0; fi

# upstream workflows must stay disabled on this repo (a merge can bring new ones in, active by default)
if command -v gh >/dev/null 2>&1; then
  act=$(gh api repos/mann1x/omk-model-optimizer/actions/workflows --paginate \
    --jq '.workflows[] | select(.state=="active") | select(.path | startswith(".github/workflows/")) | select(.path | test("/omk-") | not) | .path' 2>/dev/null)
  if [ -n "$act" ]; then echo "== ACTIVE upstream workflows (disable: gh api -X PUT repos/mann1x/omk-model-optimizer/actions/workflows/<file>/disable):"
    echo "$act" | sed 's/^/   /'; rc=1
  else echo "== upstream workflows: all disabled"; fi
fi

if [ "$(git rev-parse origin/main 2>/dev/null)" = "$(git rev-parse upstream/main)" ]; then echo "== main mirror: in sync"
else
  echo "== main mirror: origin/main $(git rev-parse --short origin/main 2>/dev/null) != upstream/main $(git rev-parse --short upstream/main)"
  if [ "${1:-}" = "--sync-main" ]; then
    git push origin upstream/main:refs/heads/main && echo "   main fast-forwarded to upstream/main" || { echo "   push failed (not a fast-forward?)"; rc=1; }
  fi
fi
exit $rc
