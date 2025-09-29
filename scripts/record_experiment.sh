#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POST="${ROOT}/scripts/post_release.sh"
RESERVE="${ROOT}/scripts/reserve_experiment.py"
FINALIZE="${ROOT}/scripts/finalize_and_publish.py"
AUTO_FINALIZE="${ROOT}/scripts/auto_finalize_experiment.py"
UPDATE_INDEX="${ROOT}/scripts/update_experiment_index.py"

usage() {
  cat <<EOF
Usage: $(basename "$0") EXP_ID [--apply] [--push] [--pr] [--reserve-if-missing]
  EXP_ID               : experiment id (e.g. EXP006G). If omitted, you will be prompted.
  --apply              : actually perform changes (default: dry-run / preview)
  --push               : when --apply, push created branch (forwarded to post_release.sh)
  --pr                 : when --apply, attempt PR creation (post_release.sh prints guidance)
  --reserve-if-missing : if experiments/EXP_ID missing, call reserve_experiment.py (interactive)
  -h, --help           : show this help
EOF
  exit 2
}

# parse args
EXP_ID="${1:-}"
shift || true
APPLY=false
DO_PUSH=false
DO_PR=false
RESERVE_IF_MISSING=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=true; shift ;;
    --push) DO_PUSH=true; shift ;;
    --pr) DO_PR=true; shift ;;
    --reserve-if-missing) RESERVE_IF_MISSING=true; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

if [[ -z "${EXP_ID}" ]]; then
  read -r -p "EXP_ID (例: EXP006G): " EXP_ID
  EXP_ID="${EXP_ID:-}"
  if [[ -z "${EXP_ID}" ]]; then
    echo "ERROR: EXP_ID required"
    exit 2
  fi
fi

echo "Workspace root: ${ROOT}"
echo "Target EXP_ID: ${EXP_ID}"
echo

# 1) If experiment dir missing, optionally reserve
if [[ ! -d "${ROOT}/experiments/${EXP_ID}" ]]; then
  echo "Notice: experiments/${EXP_ID} not found."
  if $RESERVE_IF_MISSING ; then
    read -r -p "Call reserve_experiment.py to create ${EXP_ID}? (y/N) " yn
    if [[ "${yn,,}" = "y" ]]; then
      # derive system letter as last char of EXP_ID (fallback to G)
      SYS="${EXP_ID: -1}"
      if [[ ! "${SYS}" =~ [A-Za-z] ]]; then
        SYS="G"
      fi
      python3 "${RESERVE}" --system "${SYS}" --title "${EXP_ID}" || {
        echo "reserve_experiment failed; aborting."
        exit 1
      }
      echo "Reserved ${EXP_ID}."
    else
      echo "Skipping reserve. Ensure experiments/${EXP_ID}/ exists before apply."
    fi
  else
    echo "Run with --reserve-if-missing to create a reservation, or create the directory manually."
  fi
  echo
fi

# 2) Dry-run: call post_release.sh without --apply to preview (use bash so executable bit not required)
if [[ -f "${POST}" ]]; then
  echo "=== DRY-RUN: Showing planned actions (post_release.sh) ==="
  bash "${POST}" "${EXP_ID}" || true
  echo "=== end DRY-RUN ==="
else
  echo "Warning: ${POST} not found. Please ensure ${POST} exists."
fi

# Ask user whether to proceed to apply
if ! $APPLY; then
  read -r -p "Proceed to perform changes (run with --apply)? (y/N) " cont
  if [[ "${cont,,}" != "y" ]]; then
    echo "Aborted by user (dry-run mode)."
    exit 0
  fi
fi

# 3) Collect meta inputs (allow env overrides)
GIT_COMMIT="${GIT_COMMIT:-TBD}"
OOF_AUC="${OOF_AUC:-TBD}"
PUBLIC_LB="${PUBLIC_LB:-TBD}"
SUBMISSION_ID="${SUBMISSION_ID:-TBD}"

# Attempt to auto-detect OOF AUC using finalize_and_publish helper if available
echo "Attempting to auto-detect OOF AUC from experiment artifacts..."
DETECTED_OOF=$(python3 - "${EXP_ID}" <<'PY' 2>/dev/null || echo "TBD"
import sys
try:
    from scripts.finalize_and_publish import auto_extract_oof_auc
    val = auto_extract_oof_auc(sys.argv[1])
    if val is None:
        print("TBD")
    else:
        print(f"{val}")
except Exception:
    print("TBD")
PY
)
 
if [[ -n "${DETECTED_OOF}" && "${DETECTED_OOF}" != "TBD" ]]; then
  echo "Auto-detected OOF AUC: ${DETECTED_OOF}"
  OOF_AUC="${DETECTED_OOF}"
fi

read -r -p "Git commit SHA (enter to use ${GIT_COMMIT}): " tmp && [[ -n "$tmp" ]] && GIT_COMMIT="$tmp"
read -r -p "OOF AUC (enter to use ${OOF_AUC}): " tmp && [[ -n "$tmp" ]] && OOF_AUC="$tmp"
read -r -p "Public LB (enter to use ${PUBLIC_LB}): " tmp && [[ -n "$tmp" ]] && PUBLIC_LB="$tmp"
read -r -p "Submission ID (enter to use ${SUBMISSION_ID}): " tmp && [[ -n "$tmp" ]] && SUBMISSION_ID="$tmp"

echo
echo "Will apply with:"
echo "  GIT_COMMIT=${GIT_COMMIT}"
echo "  OOF_AUC=${OOF_AUC}"
echo "  PUBLIC_LB=${PUBLIC_LB}"
echo "  SUBMISSION_ID=${SUBMISSION_ID}"
echo

# Final confirmation
read -r -p "Final confirm to run post_release.sh --apply ? (y/N) " final_ok
if [[ "${final_ok,,}" != "y" ]]; then
  echo "Aborted by user."
  exit 0
fi

# 4) Execute post_release.sh in apply mode
POST_ARGS=( "${EXP_ID}" --apply )
$DO_PUSH && POST_ARGS+=( --push )
$DO_PR && POST_ARGS+=( --pr )

# pass env vars to subprocess
export GIT_COMMIT OOF_AUC PUBLIC_LB SUBMISSION_ID

echo "=== Running post_release.sh (apply) ==="
if [[ -f "${POST}" ]]; then
  bash "${POST}" "${POST_ARGS[@]}"
else
  echo "post_release.sh not found; aborting."
  exit 1
fi

# 5) Optionally run finalize for stricter checks and PR creation
if [[ -f "${FINALIZE}" ]]; then
  echo
  read -r -p "Run finalize_and_publish.py for final checks and optional PR? (y/N) " run_fin
  if [[ "${run_fin,,}" = "y" ]]; then
    python3 "${FINALIZE}" "${EXP_ID}" "${GIT_COMMIT}" "${OOF_AUC}" "${PUBLIC_LB}" "${SUBMISSION_ID}"
  fi
else
  echo "Note: finalize_and_publish.py not found; skip finalize step."
fi

echo
echo "=== Done. ==="
echo "Review changes, run 'git status' and create PR"