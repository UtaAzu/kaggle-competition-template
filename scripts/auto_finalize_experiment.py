import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import date

TEMPLATE_REPORT = """# 実験レポート — {exp_id}
Date: {date}
Git: {git_commit}

## サマリ
- EXP_ID: {exp_id}
- CV OOF AUC: {oof_auc}
- Public LB: {public_lb}
- submission_id: {submission_id}

## 主要成果物
- models dir: {models_dir}  # Dataset slugで参照
- experiment artifacts: {experiment_dir}/artifacts

## Next action (自動補完)
- TODO: 次の一手をここに記入してください。
"""

def prompt(prompt_text, default="TBD"):
    try:
        v = input(f"{prompt_text} [{default}]: ").strip()
    except EOFError:
        return default
    return v if v else default

def write_if_missing(path: Path, content: str, force: bool) -> bool:
    if path.exists() and not force:
        print(f"SKIP (exists): {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {path}")
    return True

def copy_checklist(template: Path, dest: Path, force: bool) -> bool:
    if not template.exists():
        dest.write_text("- [ ] review artifacts\n", encoding="utf-8")
        print(f"Wrote {dest} (minimal)")
        return True
    if dest.exists() and not force:
        print(f"SKIP (exists): {dest}")
        return False
    dest.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {dest} (from template)")
    return True

def parse_args():
    p = argparse.ArgumentParser(description="Auto finalize experiment (safe, non-destructive by default)")
    p.add_argument("exp_id", nargs="?", help="EXP ID (e.g. EXP006G)")
    p.add_argument("git_commit", nargs="?", help="Git commit SHA or TBD")
    p.add_argument("oof_auc", nargs="?", help="OOF AUC or TBD")
    p.add_argument("public_lb", nargs="?", help="Public LB or TBD")
    p.add_argument("submission_id", nargs="?", help="Submission ID or TBD")
    p.add_argument("--force", action="store_true", help="Overwrite existing files")
    p.add_argument("--no-index", action="store_true", help="Skip running update_experiment_index.py")
    return p.parse_args()

def main():
    args = parse_args()
    if args.exp_id:
        exp_id = args.exp_id
        git_commit = args.git_commit or "TBD"
        oof_auc = args.oof_auc or "TBD"
        public_lb = args.public_lb or "TBD"
        submission_id = args.submission_id or "TBD"
    else:
        print("Auto finalize experiment (GUI-friendly). You can run via VSCode ▶ Run Python File.")
        exp_id = prompt("EXP_ID (例: EXP005G_cv_audit)", "EXP_TODO")
        git_commit = prompt("Git commit SHA", "TBD")
        oof_auc = prompt("OOF AUC", "TBD")
        public_lb = prompt("Public LB", "TBD")
        submission_id = prompt("Submission ID", "TBD")

    root = Path(".").resolve()
    exp_dir = root / "experiments" / exp_id
    models_dir = root / "artifacts" / "models" / exp_id  # 廃止予定: Dataset slugで参照
    exp_artifacts = exp_dir / "artifacts"
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_artifacts.mkdir(parents=True, exist_ok=True)

    # CV-LBギャップ分析に対応したrun.json構造
    run = {
        "experiment_id": exp_id,
        "title": exp_id,
        "date": date.today().isoformat(),
        "status": "completed",
        "git_commit": git_commit,
        "cv": {
            "oof_auc": float(oof_auc) if oof_auc not in ("TBD", None, "") else None
        },
        "leaderboard": {
            "public_lb": float(public_lb) if public_lb not in ("TBD", None, "") else None,
            "submission_id": submission_id
        },
        "artifacts": {
            "models_dir": str(models_dir),
            "experiment_dir": str(exp_dir),
            "dataset_uri": None,
            "models_location": "kaggle_dataset"
        },
        "model_dataset_slug": None  # finalize_and_publish.pyで対話式追加
    }

    # 非破壊作成
    write_if_missing(exp_dir / "run.json", json.dumps(run, ensure_ascii=False, indent=2), force=args.force)

    report = TEMPLATE_REPORT.format(exp_id=exp_id, date=date.today().isoformat(), git_commit=git_commit,
                                    oof_auc=oof_auc, public_lb=public_lb, submission_id=submission_id,
                                    models_dir=str(models_dir), experiment_dir=str(exp_dir))
    write_if_missing(exp_dir / "report.md", report, force=args.force)

    readme = f"# {exp_id} (Run Card)\n\n- run.json: ./run.json\n- report: ./report.md\n\nShort: OOF AUC={oof_auc}, Public LB={public_lb}\n"
    write_if_missing(exp_dir / "README.md", readme, force=args.force)

    checklist_src = Path("experiments/_template/CHECKLIST.md")
    copy_checklist(checklist_src, exp_dir / "checklist.md", force=args.force)

    # index update optional
    if not args.no_index:
        ans = "n"
        try:
            ans = input("Run scripts/update_experiment_index.py now? (y/n) [n]: ").strip().lower()
        except EOFError:
            ans = "n"
        if ans == "y":
            try:
                subprocess.run([sys.executable, "scripts/update_experiment_index.py"], check=True)
                print("Updated experiments/INDEX.md")
            except Exception as e:
                print("Index update failed:", e)
                print("Run `python scripts/update_experiment_index.py` later.")
        else:
            print("Skipped index update.")

    print("Done. Review files and commit/push.")

if __name__ == "__main__":
    main()