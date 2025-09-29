#!/usr/bin/env bash
# Lightweight post-release checklist + optional apply mode for experiments
# Usage (dry-run):   ./scripts/post_release.sh EXP_ID
# Usage (apply):     ./scripts/post_release.sh EXP_ID --apply [--push] [--pr]
# Env vars (optional): GIT_COMMIT, OOF_AUC, PUBLIC_LB, SUBMISSION_ID, GITHUB_TOKEN, GITHUB_REPO
set -euo pipefail

EXP_ID="${1:-}"
shift || true

APPLY=false
DO_PUSH=false
DO_PR=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=true; shift ;;
    --push) DO_PUSH=true; shift ;;
    --pr) DO_PR=true; shift ;;
    --help|-h) echo "Usage: $0 EXP_ID [--apply] [--push] [--pr]"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

if [[ -z "$EXP_ID" ]]; then
  echo "ERROR: EXP_ID required"
  echo "Usage: $0 EXP_ID [--apply] [--push] [--pr]"
  exit 2
fi

GIT_COMMIT="${GIT_COMMIT:-TBD}"
OOF_AUC="${OOF_AUC:-TBD}"
PUBLIC_LB="${PUBLIC_LB:-TBD}"
SUBMISSION_ID="${SUBMISSION_ID:-TBD}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Post-release checklist for ${EXP_ID}"
echo
echo "1) 非破壊で自動生成 (run.json/report/README/checklist)"
echo "   python ${ROOT}/scripts/auto_finalize_experiment.py ${EXP_ID} ${GIT_COMMIT} ${OOF_AUC} ${PUBLIC_LB} ${SUBMISSION_ID} --no-index"
echo "   (参照: scripts/auto_finalize_experiment.py)"
echo
echo "2) インデックス再生成 (Codespaces / ローカル)"
echo "   python ${ROOT}/scripts/update_experiment_index.py"
echo
echo "3) スコアボードに追記 (docs/scoreboard.md)"
echo "   (append if not exists) '### ${EXP_ID}' block with OOF/Public/Date"
echo
echo "4) ファイルをステージしてコミット / push"
echo "   git checkout -b docs/${EXP_ID}-record"
echo "   git add experiments/${EXP_ID} docs/scoreboard.md experiments/INDEX.md"
echo "   git commit -m \"${EXP_ID}: record results and scoreboard\""
echo "   git push -u origin HEAD"
echo
echo "5) PR 作成（手動 or CI）"
echo "   - 手動: コピーした PR ボディを GitHub 上で作成"
echo "   - 自動化する場合は GITHUB_TOKEN と GITHUB_REPO を安全に設定 (参照: scripts/finalize_and_publish.py)"
echo

if ! $APPLY; then
  echo "Dry-run only. Re-run with --apply to execute the commands above."
  exit 0
fi

# === APPLY MODE ===
echo "=== Running auto_finalize_experiment.py (safe mode: --no-index) ==="
if ! python3 "${ROOT}/scripts/auto_finalize_experiment.py" "${EXP_ID}" "${GIT_COMMIT}" "${OOF_AUC}" "${PUBLIC_LB}" "${SUBMISSION_ID}" --no-index; then
  echo "Warning: auto_finalize_experiment.py failed."
  if $APPLY; then
    read -r -p "Continue despite failure? (y/N) " yn
    if [[ "${yn,,}" != "y" ]]; then
      echo "Aborting as requested."
      exit 1
    fi
  else
    echo "Dry-run: continue. Re-run with --apply to prompt for action."
  fi
fi

echo "=== Updating experiments/INDEX.md ==="
if ! python3 "${ROOT}/scripts/update_experiment_index.py"; then
  echo "Warning: update_experiment_index.py failed; continuing"
fi

# append to docs/scoreboard.md if not present
SCOREBOARD="${ROOT}/docs/scoreboard.md"
DATE="$(python3 - <<PY
from datetime import date
print(date.today().isoformat())
PY
)"
SB_ENTRY="### ${EXP_ID}\n- OOF AUC: ${OOF_AUC}\n- Public LB: ${PUBLIC_LB}\n- Date: ${DATE}\n"

mkdir -p "$(dirname "${SCOREBOARD}")"
if [[ -f "${SCOREBOARD}" ]]; then
  # 厳密に "### <EXP_ID>" 見出しを探す（部分一致を避ける）
  if grep -E "^###[[:space:]]+${EXP_ID}\\b" "${SCOREBOARD}" >/dev/null 2>&1; then
    echo "Scoreboard already contains ${EXP_ID} — skipping append"
  else
    printf "\n%s\n" "${SB_ENTRY}" >> "${SCOREBOARD}"
    echo "Appended scoreboard entry to ${SCOREBOARD}"
  fi
else
  printf "# Experiment Scoreboard\n\n%s\n" "${SB_ENTRY}" > "${SCOREBOARD}"
  echo "Created ${SCOREBOARD}"
fi

# Prepare files to add
BRANCH="docs/${EXP_ID}-record"
FILES_TO_ADD=()
if [[ -d "${ROOT}/experiments/${EXP_ID}" ]]; then
  FILES_TO_ADD+=("experiments/${EXP_ID}")
fi
if [[ -f "${SCOREBOARD}" ]]; then
  FILES_TO_ADD+=("docs/scoreboard.md")
fi
if [[ -f "${ROOT}/experiments/INDEX.md" ]]; then
  FILES_TO_ADD+=("experiments/INDEX.md")
fi

# README.md は自動追加しない（誤コミット防止）。必要なら手動で追加するよう促す。
echo "Note: README.md is not auto-added. Run 'git add README.md' manually if you updated it."

if [[ ${#FILES_TO_ADD[@]} -eq 0 ]]; then
  echo "No files detected to add/commit. Ensure experiments/${EXP_ID}/ contains outputs."
  exit 0
fi

echo "Files to add: ${FILES_TO_ADD[*]}"
read -r -p "Proceed with git branch/commit? (y/N) " yn
if [[ "${yn,,}" != "y" ]]; then
  echo "Skipping git commit/push as requested."
  exit 0
fi

# git operations
echo "Creating branch ${BRANCH} and committing files..."
git checkout -B "${BRANCH}"
git add "${FILES_TO_ADD[@]}"
if git commit -m "${EXP_ID}: record results and scoreboard"; then
  echo "Committed."
else
  echo "Nothing to commit (or commit failed)."
fi

if $DO_PUSH; then
  echo "Pushing branch ${BRANCH}..."
  git push -u origin "${BRANCH}"
else
  echo "Push skipped (run with --push to push)."
fi

# PR body (print)
PR_TITLE="${EXP_ID}: record experiment results"
PR_BODY=$(cat <<EOF
Auto-generated run card and scoreboard update for ${EXP_ID}.

CV OOF AUC: ${OOF_AUC}
Public LB: ${PUBLIC_LB}
Submission ID: ${SUBMISSION_ID}

Files to review:
- experiments/${EXP_ID}
- docs/scoreboard.md
- experiments/INDEX.md

See: experiments/${EXP_ID}/run.json
EOF
)

echo
echo "=== PR title ==="
echo "${PR_TITLE}"
echo
echo "=== PR body ==="
echo "${PR_BODY}"
echo

if $DO_PR; then
  if [[ -z "${GITHUB_TOKEN:-}" || -z "${GITHUB_REPO:-}" ]]; then
    echo "GITHUB_TOKEN and GITHUB_REPO must be set to auto-create PR. Skipping."
  else
    echo "Auto PR creation is intentionally NOT implemented in this script to avoid token leakage."
    echo "Use scripts/finalize_and_publish.py or the GitHub UI to create the PR. Example:"
    echo "  curl -H \"Authorization: token \$GITHUB_TOKEN\" -X POST -d '{\"title\":\"${PR_TITLE}\",\"body\":\"${PR_BODY}\",\"head\":\"${BRANCH}\",\"base\":\"main\"}' https://api.github.com/repos/\$GITHUB_REPO/pulls"
  fi
fi

echo "Done."