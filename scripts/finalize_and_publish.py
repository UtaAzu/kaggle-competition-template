#!/usr/bin/env python3
"""
CV-LBギャップ調査を前提としたrun.json標準化＆実験記録自動化スクリプト
- oof_auc, public_lb, submission_idは自動抽出を優先し、失敗時のみ手入力
- 抽出ロジックはextract_metrics.pyからimport（なければ内部関数でfallback）
- 必須: run.json生成・標準化（experiment_id, date, git_commit, cv.oof_auc, leaderboard.public_lb, model_dataset_slug, artifacts.dataset_uri, artifacts.models_location）
- 対話式: model_dataset_slug（Kaggle Dataset slug）を必ず入力
- 必須ファイルチェック: run.jsonと主要フィールドのみ（軽量成果物は推奨）
- scoreboard追記: docs/scoreboard.md
- git/PR補助: branch作成・commit/push・PR作成（GITHUB_TOKEN/GITHUB_REPOがあれば自動）

Usage:
  python scripts/finalize_and_publish.py
  python scripts/finalize_and_publish.py EXP_ID GIT_COMMIT OOF_AUC PUBLIC_LB SUBMISSION_ID
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import date
from typing import List

ROOT = Path(".").resolve()

# --- 自動抽出ロジック ---
def try_import_extract_metrics():
    try:
        import extract_metrics
        return extract_metrics
    except ImportError:
        return None

def auto_extract_oof_auc(exp_id: str) -> str | None:
    # metrics.json優先, oof.csv fallback
    exp_dir = ROOT / "experiments" / exp_id / f"{exp_id.lower().replace('_', '-')}-artifacts"
    metrics_path = exp_dir / "metrics.json"
    if metrics_path.exists():
        try:
            d = json.loads(metrics_path.read_text(encoding="utf-8"))
            for k in ("oof_auc", "mean_auc", "auc", "mean"):
                if k in d:
                    try:
                        return str(float(d[k]))
                    except Exception:
                        continue
        except Exception:
            pass
    oof_path = exp_dir / "oof.csv"
    if oof_path.exists():
        try:
            import pandas as pd
            from sklearn.metrics import roc_auc_score
            df = pd.read_csv(oof_path)
            # 推定カラム
            pred_col = "oof_pred" if "oof_pred" in df.columns else ("rule_violation" if "rule_violation" in df.columns else None)
            label_col = "rule_violation"
            if pred_col and label_col in df.columns and df[label_col].nunique() >= 2:
                auc = roc_auc_score(df[label_col].values, df[pred_col].values)
                return str(round(float(auc), 6))
        except Exception:
            pass
    return None

def auto_extract_public_lb(exp_id: str) -> str | None:
    # run.json, metrics.json, submission.csv
    exp_dir = ROOT / "experiments" / exp_id
    run_path = exp_dir / "run.json"
    if run_path.exists():
        try:
            d = json.loads(run_path.read_text(encoding="utf-8"))
            lb = d.get("leaderboard", {})
            if isinstance(lb, dict) and "public_lb" in lb and lb["public_lb"] not in (None, "TBD"):
                return str(lb["public_lb"])
        except Exception:
            pass
    metrics_path = exp_dir / "artifacts" / "metrics.json"
    if metrics_path.exists():
        try:
            d = json.loads(metrics_path.read_text(encoding="utf-8"))
            for k in ("public_lb", "lb", "public_score"):
                if k in d and isinstance(d[k], (float, int)):
                    return str(d[k])
        except Exception:
            pass
    return None

def auto_extract_submission_id(exp_id: str) -> str | None:
    # run.json
    exp_dir = ROOT / "experiments" / exp_id
    run_path = exp_dir / "run.json"
    if run_path.exists():
        try:
            d = json.loads(run_path.read_text(encoding="utf-8"))
            lb = d.get("leaderboard", {})
            if isinstance(lb, dict) and "submission_id" in lb and lb["submission_id"] not in (None, "TBD"):
                return str(lb["submission_id"])
        except Exception:
            pass
    return None

def run_cmd(cmd: List[str], check: bool = True, capture_output: bool = False):
    print(f"> {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=True)

def prompt(text: str, default: str = "TBD") -> str:
    try:
        v = input(f"{text} [{default}]: ").strip()
    except EOFError:
        return default
    return v if v else default

def _is_float(s) -> bool:
    try:
        if s is None:
            return False
        float(s)
        return True
    except Exception:
        return False

def ensure_run_json(exp_id: str, git_commit: str, oof_auc: str, public_lb: str, submission_id: str) -> Path:
    exp_dir = ROOT / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    run_path = exp_dir / "run.json"
    run_data = {}
    if run_path.exists():
        try:
            loaded = json.loads(run_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                run_data = dict(loaded)
        except Exception:
            run_data = {}
    run_data.setdefault("experiment_id", exp_id)
    run_data.setdefault("title", exp_id)
    run_data.setdefault("date", date.today().isoformat())
    run_data.setdefault("status", "completed")
    if "artifacts" not in run_data or not isinstance(run_data.get("artifacts"), dict):
        run_data["artifacts"] = {"models_dir": f"artifacts/models/{exp_id}", "experiment_dir": str(exp_dir)}
    if git_commit and git_commit != "TBD":
        run_data["git_commit"] = git_commit
    if not isinstance(run_data.get("cv"), dict):
        run_data["cv"] = {}
    cv_dict = run_data["cv"]
    if _is_float(oof_auc):
        v = float(oof_auc)
        cv_dict["mean"] = v
        cv_dict["oof_auc"] = v
        cv_dict["mean_auc"] = v  # 互換性のため
    if not isinstance(run_data.get("leaderboard"), dict):
        run_data["leaderboard"] = {}
    lb_dict = run_data["leaderboard"]
    if _is_float(public_lb):
        lb_dict["public_lb"] = float(public_lb)
    if submission_id and submission_id != "TBD":
        lb_dict["submission_id"] = submission_id
    try:
        run_path.write_text(json.dumps(run_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {run_path}")
    except Exception as e:
        print(f"ERROR: run.json書き込み失敗: {e}")
        raise
    # === 修正: Kaggle Notebook情報を追加 ===
    kaggle_notebook = prompt("Kaggle Notebook名 (例: mabe-linear-models)", run.get("kaggle_notebook", "TBD"))
    kaggle_version = prompt("Kaggle Version (例: 1)", str(run.get("kaggle_version", "TBD")))
    
    run["kaggle_notebook"] = kaggle_notebook
    run["kaggle_version"] = int(kaggle_version) if str(kaggle_version).isdigit() else None
    
    try:
        run_path.write_text(json.dumps(run_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {run_path}")
    except Exception as e:
        print(f"ERROR: run.json書き込み失敗: {e}")
        raise
    return run_path

def interactive_dataset_input(run_path: Path):
    try:
        run_data = json.loads(run_path.read_text(encoding="utf-8"))
    except Exception:
        print(f"ERROR: run.json読み取り失敗: {run_path}")
        return
    if not run_data.get("model_dataset_slug"):
        print("\n=== Kaggle Model Dataset slug を入力してください ===")
        slug = prompt("Model dataset slug (例: username/<dataset>-exp004d-dataset)", "")
        while not slug or slug in ("TBD", ""):
            print("⚠️  Model dataset slugは必須です。")
            slug = prompt("Model dataset slug", "")
        run_data["model_dataset_slug"] = slug
        if "artifacts" not in run_data:
            run_data["artifacts"] = {}
        run_data["artifacts"]["dataset_uri"] = f"kaggle://{slug}"
        run_data["artifacts"]["models_location"] = "kaggle_dataset"
        try:
            run_path.write_text(json.dumps(run_data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"✅ Dataset情報を {run_path} に追加しました")
        except Exception as e:
            print(f"ERROR: Dataset情報の保存に失敗: {e}")
    else:
        print(f"✅ model_dataset_slug: {run_data['model_dataset_slug']} (run.jsonに既に存在)")

def check_required_files(exp_id: str):
    warnings = []
    errors = []
    exp_dir = ROOT / "experiments" / exp_id
    exp_artifacts = exp_dir / "artifacts"
    run_json = exp_dir / "run.json"
    if not run_json.exists():
        errors.append(f"{run_json} が存在しません。先にrun.jsonを作成してください。")
    else:
        try:
            run_data = json.loads(run_json.read_text(encoding="utf-8"))
            for field in ["experiment_id", "date"]:
                if field not in run_data:
                    errors.append(f"{run_json} (missing required field: {field})")
            if not run_data.get("model_dataset_slug") and not run_data.get("artifacts", {}).get("dataset_uri"):
                warnings.append(f"{run_json}にmodel_dataset_slug/dataset_uriがありません（Kaggle Dataset slugを推奨）")
            if not run_data.get("git_commit"):
                warnings.append(f"{run_json}にgit_commitがありません（再現性のため推奨）")
        except Exception as e:
            errors.append(f"{run_json}の読み取りに失敗: {e}")
    # 編集ポイント: 新成果物を追加する場合はここに推奨ファイルを追記
    if exp_artifacts.exists():
        recommended_files = ["submission.csv", "oof.csv", "metrics.json"]
        found_any = any((exp_artifacts / f).exists() for f in recommended_files)
        if not found_any:
            nested_found = bool(list(exp_artifacts.rglob("*.csv")) or list(exp_artifacts.rglob("*.json")))
            if not nested_found:
                warnings.append(f"{exp_artifacts}に軽量成果物がありません: {recommended_files}")
    else:
        warnings.append(f"{exp_artifacts}ディレクトリが存在しません。軽量成果物の配置を推奨します。")
    if warnings:
        print("⚠️  警告:")
        for w in warnings:
            print(f"   - {w}")
    if errors:
        print("❌ 必須エラー:")
        for e in errors:
            print(f"   - {e}")
        sys.exit(1)
    print("✅ 必須ファイルチェック完了")

def append_scoreboard(exp_id: str, oof_auc: str, public_lb: str) -> Path:
    sb = ROOT / "docs" / "scoreboard.md"
    line = f"\n### {exp_id}\n- OOF AUC: {oof_auc}\n- Public LB: {public_lb}\n- Date: {date.today().isoformat()}\n"
    if not sb.exists():
        (sb.parent).mkdir(parents=True, exist_ok=True)
        sb.write_text("# Experiment Scoreboard\n\n" + line, encoding="utf-8")
        print(f"created {sb}")
        return sb
    text = sb.read_text(encoding="utf-8")
    if exp_id in text:
        print(f"{exp_id} already in docs/scoreboard.md — skipping append")
        return sb
    with sb.open("a", encoding="utf-8") as f:
        f.write(line)
    print(f"appended summary to {sb}")
    return sb

def git_commit_and_push(paths: List[str], branch: str, message: str):
    run_cmd(["git", "checkout", "-B", branch])
    run_cmd(["git", "add"] + paths)
    try:
        run_cmd(["git", "commit", "-m", message])
    except subprocess.CalledProcessError as e:
        print("git commit failed (possibly nothing to commit):", e)
    run_cmd(["git", "push", "-u", "origin", branch])

def create_github_pr(repo: str, token: str, title: str, body: str, head: str, base: str = "main"):
    payload = {"title": title, "body": body, "head": head, "base": base}
    url = f"https://api.github.com/repos/{repo}/pulls"
    try:
        import requests
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json"
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        if resp.status_code not in (200, 201):
            print(f"PR creation failed: HTTP {resp.status_code} - {resp.text}")
            return None
        j = resp.json()
        print("PR created:", j.get("html_url"))
        return j.get("html_url")
    except Exception as e:
        print("Warning: 'requests' not available, falling back to curl")
        cmd = [
            "curl", "-s",
            "-H", f"Authorization: token {token}",
            "-H", "Accept: application/vnd.github+json",
            "-d", json.dumps(payload),
            url
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            try:
                print("PR creation failed (curl):", res.stderr.decode(errors="ignore"))
            except Exception as e2:
                print("PR creation failed (curl):", str(e2))
            return None
        out = res.stdout.decode(errors="ignore")
        try:
            j = json.loads(out)
            print("PR created:", j.get("html_url"))
            return j.get("html_url")
        except Exception as e:
            print("PR response parse failed (curl):", e)
            print(out)
            return None

def main():
    extract_metrics = try_import_extract_metrics()
    if len(sys.argv) >= 6:
        exp_id, git_commit, oof_auc, public_lb, submission_id = sys.argv[1:6]
    else:
        exp_id = prompt("EXP_ID (例: EXP005G)", "EXP_TODO")
        git_commit = prompt("git commit SHA (空で自動検出)", "TBD")
        if git_commit in ("", "TBD"):
            try:
                sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
                git_commit = sha
            except Exception:
                git_commit = "TBD"
        # --- 自動抽出 ---
        oof_auc = None
        public_lb = None
        submission_id = None
        if extract_metrics:
            oof_auc = extract_metrics.auto_extract_oof_auc(exp_id)
            public_lb = extract_metrics.auto_extract_public_lb(exp_id)
            submission_id = extract_metrics.auto_extract_submission_id(exp_id)
        else:
            oof_auc = auto_extract_oof_auc(exp_id)
            public_lb = auto_extract_public_lb(exp_id)
            submission_id = auto_extract_submission_id(exp_id)
        print(f"[auto] OOF AUC: {oof_auc}")
        print(f"[auto] Public LB: {public_lb}")
        print(f"[auto] Submission ID: {submission_id}")
        # --- 手入力fallback ---
        oof_auc = prompt("OOF AUC", oof_auc or "TBD")
        public_lb = prompt("Public LB", public_lb or "TBD")
        submission_id = prompt("Submission ID", submission_id or "TBD")
    run_path = ensure_run_json(exp_id, git_commit, oof_auc, public_lb, submission_id)
    interactive_dataset_input(run_path)
    check_required_files(exp_id)
    append_scoreboard(exp_id, oof_auc, public_lb)
    branch = f"docs/{exp_id}-record"
    files_to_add = []
    exp_dir = ROOT / "experiments" / exp_id
    if exp_dir.exists():
        files_to_add.append(str(exp_dir))
    if (ROOT / "docs" / "scoreboard.md").exists():
        files_to_add.append("docs/scoreboard.md")
    if (ROOT / "experiments" / "INDEX.md").exists():
        files_to_add.append("experiments/INDEX.md")
    if not files_to_add:
        print("No files to add/commit. Ensure experiments/<EXP_ID>/ contains files.")
        return
    print("About to create git branch and commit the following paths:", files_to_add)
    proceed = prompt("Proceed with git commit & push? (y/n)", "n")
    if proceed.lower() == "y":
        try:
            git_commit_and_push(files_to_add, branch, f"{exp_id}: record results and scoreboard")
        except Exception as e:
            print("git push failed:", e)
            print("Please run git commands manually.")
    else:
        print("Skipping git commit/push. You can run these commands manually.")
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")
    if token and repo and proceed.lower() == "y":
        title = f"{exp_id}: record experiment results"
        body = f"Auto-generated run card and scoreboard update for {exp_id}.\n\nCV OOF AUC: {oof_auc}\nPublic LB: {public_lb}\nSubmission: {submission_id}\n\nSee experiments/{exp_id}."
        pr_url = create_github_pr(repo, token, title, body, head=branch, base="main")
        if not pr_url:
            print("PR creation failed. Provide GITHUB_TOKEN/GITHUB_REPO or create PR manually.")
    else:
        pr_body = f"# PR: {exp_id} record\n\nCV OOF AUC: {oof_auc}\nPublic LB: {public_lb}\nSubmission ID: {submission_id}\n\nFiles to review: experiments/{exp_id}, docs/scoreboard.md, experiments/INDEX.md\n"
        print("\n--- PR body (copy/paste to GitHub) ---\n")
        print(pr_body)
        print("\n--- End ---\n")
    print("Done. Next: run `.prompts/01_generate_report.md` and `.prompts/06_devils_advocate_review.md` prompts (use experiments/{exp_id}/run.json as Primary Resource).")
    print("References: .prompts/08_post_experiment_report.md, .prompts/01_generate_report.md, .prompts/06_devils_advocate_review.md")

if __name__ == "__main__":
    main()