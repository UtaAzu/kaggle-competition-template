#!/usr/bin/env python3
"""
Scan a Kaggle artifacts directory for an experiment and record lightweight metadata
into experiments/<EXP_ID>/artifacts/metadata_from_kaggle.json and experiments/<EXP_ID>/run.json.

Usage:
  python3 scripts/record_artifacts_kaggle.py EXP_ID --slug owner/dataset-name \
      --kaggle-artifacts-path /kaggle/input/jigsaw-exp001d-dataset/artifacts

Options:
  --max-bytes N    : skip hashing files larger than N bytes (0 = hash all). Default 0.
  --skip-extensions ext1,ext2 : comma separated list of extensions to skip hashing (e.g. .bin,.safetensors)
"""
import argparse
import json
import hashlib
from pathlib import Path
import sys
from typing import List, Dict, Any

ROOT = Path(".").resolve()


def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_artifacts(base: Path, max_bytes: int = 0, skip_exts: List[str] = None) -> List[Dict[str, Any]]:
    files = []
    skip_exts = skip_exts or []
    if not base.exists():
        raise FileNotFoundError(f"Kaggle artifacts path not found: {base}")
    for p in sorted(base.rglob("*")):
        if p.is_file():
            rel = p.relative_to(ROOT) if str(p).startswith(str(ROOT)) else p
            size = p.stat().st_size
            ext = p.suffix.lower()
            should_hash = True
            if max_bytes and size > max_bytes:
                should_hash = False
            if ext and ext in skip_exts:
                should_hash = False
            sha256 = None
            try:
                if should_hash:
                    sha256 = sha256_of_file(p)
            except Exception as e:
                sha256 = f"ERROR: {e}"
            files.append({
                "path": str(p),
                "relpath": str(rel),
                "size": size,
                "sha256": sha256,
            })
    return files


def load_run_json(exp_dir: Path) -> Dict[str, Any]:
    run_json = exp_dir / "run.json"
    if run_json.exists():
        try:
            return json.loads(run_json.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_run_json(exp_dir: Path, data: Dict[str, Any]):
    run_json = exp_dir / "run.json"
    run_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated run.json: {run_json}")


def main(argv: List[str] = None):
    p = argparse.ArgumentParser(description="Record Kaggle artifacts metadata into experiments/<EXP>/artifacts")
    p.add_argument("exp_id")
    p.add_argument("--slug", required=True, help="Kaggle dataset slug (owner/dataset-name)")
    p.add_argument("--kaggle-artifacts-path", default="/kaggle/input", help="Base path where Kaggle inputs live (default /kaggle/input)")
    p.add_argument("--max-bytes", type=int, default=0, help="Skip hashing files larger than this many bytes (0 = hash all)")
    p.add_argument("--skip-extensions", default="", help="Comma-separated list of extensions to skip hashing, e.g. .safetensors,.bin")
    args = p.parse_args(argv)

    exp_id = args.exp_id
    slug = args.slug
    base = Path(args.kaggle_artifacts_path)
    # allow passing the full path including dataset and artifacts
    if (base / "artifacts" / exp_id).exists():
        artifacts_dir = base / "artifacts" / exp_id
    elif base.exists() and (base / exp_id).exists():
        artifacts_dir = base / exp_id
    else:
        # if user passed full path like /kaggle/input/jigsaw-exp001d-dataset/artifacts
        # try to use it directly if it contains exp_id
        if str(base).endswith(str(exp_id)) and base.exists():
            artifacts_dir = base
        else:
            # try common pattern: /kaggle/input/<slug>/artifacts/<exp_id>
            candidate = Path("/kaggle/input") / slug.split("/")[-1] / "artifacts" / exp_id
            if candidate.exists():
                artifacts_dir = candidate
            else:
                # fallback: if explicit path includes 'artifacts' check directly
                direct = Path(args.kaggle_artifacts_path)
                if direct.exists():
                    artifacts_dir = direct
                else:
                    print(f"Error: could not locate artifacts directory for exp {exp_id} under {base}")
                    sys.exit(2)

    skip_exts = [e.strip().lower() for e in args.skip_extensions.split(",") if e.strip()]
    print(f"Collecting artifacts from: {artifacts_dir}")
    files = collect_artifacts(artifacts_dir, max_bytes=args.max_bytes, skip_exts=skip_exts)

    exp_dir = ROOT / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = exp_dir / "artifacts"
    meta_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_dataset_slug": slug,
        "artifact_dataset_uri": f"kaggle://{slug}/artifacts/{exp_id}",
        "artifact_source_path": str(artifacts_dir),
        "artifact_files": files,
    }

    metadata_path = meta_dir / "metadata_from_kaggle.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote metadata: {metadata_path} ({len(files)} files)")

    # update run.json with artifacts info
    run_data = load_run_json(exp_dir)
    if not isinstance(run_data, dict):
        run_data = {}
    if "artifacts" not in run_data or not isinstance(run_data.get("artifacts"), dict):
        run_data["artifacts"] = {}
    run_data["artifacts"].update({
        "model_dataset_slug": slug,
        "artifact_dataset_uri": f"kaggle://{slug}/artifacts/{exp_id}",
        "artifact_source_path": str(artifacts_dir)
    })
    # optionally store summary counts/sizes
    total_size = sum(f["size"] for f in files)
    run_data["artifacts"]["file_count"] = len(files)
    run_data["artifacts"]["total_bytes"] = total_size

    save_run_json(exp_dir, run_data)


if __name__ == "__main__":
    main()
