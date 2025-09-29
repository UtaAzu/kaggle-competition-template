# python
# filepath: /workspaces/Kaggle-Jigsaw2025/scripts/reserve_experiment.py
#!/usr/bin/env python3
"""
Experiment Reservation Tool (拡張版)

主な追加:
- experiments/.last_exp_number に最後に割り当てた番号を永続化し、
  ディレクトリを削除しても欠番にならないようにする。

Usage:
    python scripts/reserve_experiment.py --system G --title "my_experiment_title"
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
TEMPLATE_DIR = EXPERIMENTS_DIR / "_template"
CONFIG_DIR = ROOT / "config" / "experiments"
ARTIFACTS_DIR = ROOT / "artifacts" / "models"

VALID_SYSTEMS = ["G", "D", "E", "T"]  # G=Linear, D=Deep, E=Ensemble, T=Test

# 永続化ファイル（experiment ディレクトリ直下に置く）
LAST_NUMBER_FILE = EXPERIMENTS_DIR / ".last_exp_number"


def _read_last_assigned() -> int:
    try:
        if LAST_NUMBER_FILE.exists():
            txt = LAST_NUMBER_FILE.read_text(encoding="utf-8").strip()
            if txt.isdigit():
                return int(txt)
    except Exception:
        pass
    return 0


def _write_last_assigned(n: int) -> None:
    try:
        LAST_NUMBER_FILE.parent.mkdir(parents=True, exist_ok=True)
        LAST_NUMBER_FILE.write_text(str(n), encoding="utf-8")
    except Exception:
        pass


def find_next_experiment_id(system: str) -> str:
    """Find the next available experiment ID for the given system."""
    existing_experiments = []
    if EXPERIMENTS_DIR.exists():
        for exp_dir in EXPERIMENTS_DIR.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.startswith("_"):
                name = exp_dir.name
                if name.startswith("EXP") and system in name:
                    try:
                        number_part = name[3:6]  # EXP###X -> ### を抜く
                        if number_part.isdigit():
                            existing_experiments.append(int(number_part))
                    except (ValueError, IndexError):
                        continue

    max_existing = max(existing_experiments) if existing_experiments else 0
    last_assigned = _read_last_assigned()
    next_number = max(max_existing, last_assigned) + 1
    return f"EXP{next_number:03d}{system}"


def create_readme_template(exp_id: str, title: str, system: str) -> str:
    return f"# {exp_id} (Run Card)\n\n- Title: {title}\n- System: {system}\n\nShort: reserved {datetime.now().isoformat()}\n"


def create_run_metadata(exp_id: str, title: str, system: str) -> Dict[str, Any]:
    return {
        "experiment_id": exp_id,
        "title": title,
        "system": system,
        "created_at": datetime.now().isoformat(),
        "status": "reserved",
        "git_commit": None,
        "config_path": f"config/experiments/{exp_id}.yaml",
        "artifacts": {
            "models_dir": f"artifacts/models/{exp_id}",
            "experiment_dir": f"experiments/{exp_id}"
        },
        "cv": {},
        "results": {},
        "notes": ""
    }


def create_minimal_config(config_path: Path, exp_id: str) -> None:
    minimal = {
        "experiment_name": exp_id,
        "artifacts": {"experiment_name": exp_id, "models_dir": "artifacts/models", "preprocessors_dir": "artifacts/preprocessors"},
        "n_splits": 5,
        "random_state": 42
    }
    import yaml
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(minimal, f, default_flow_style=False, sort_keys=False)


def create_config_template(exp_id: str, system: str, base_template: Optional[str] = None) -> Path:
    """Create configuration template for the experiment."""
    config_path = CONFIG_DIR / f"{exp_id}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if base_template:
        base_config_path = CONFIG_DIR / f"{base_template}.yaml"
        if not base_config_path.exists():
            base_config_path = ROOT / "config" / f"{base_template}.yaml"

        if base_config_path.exists():
            import yaml
            with open(base_config_path, 'r', encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            if 'artifacts' not in config:
                config['artifacts'] = {}
            config['artifacts']['experiment_name'] = exp_id
            config['artifacts']['models_dir'] = f"artifacts/models/{exp_id}"
            config['artifacts']['preprocessors_dir'] = f"artifacts/preprocessors/{exp_id}"
            with open(config_path, 'w', encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            return config_path
        else:
            create_minimal_config(config_path, exp_id)
            return config_path
    else:
        create_minimal_config(config_path, exp_id)
        return config_path


def create_experiment_structure(exp_id: str, title: str, system: str, template: Optional[str] = None) -> Path:
    """Create the experiment directory structure."""
    exp_dir = EXPERIMENTS_DIR / exp_id
    if exp_dir.exists():
        raise ValueError(f"Experiment directory already exists: {exp_dir}")

    # Create experiment directory and subdirs
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "notebooks").mkdir(exist_ok=True)
    (exp_dir / "artifacts").mkdir(exist_ok=True)

    # Copy template files if available
    if template and TEMPLATE_DIR.exists():
        for item in TEMPLATE_DIR.iterdir():
            if item.is_file():
                destination = exp_dir / item.name
                if not destination.exists():
                    shutil.copy2(item, destination)

    # README and run.json
    readme_content = create_readme_template(exp_id, title, system)
    (exp_dir / "README.md").write_text(readme_content, encoding="utf-8")

    run_metadata = create_run_metadata(exp_id, title, system)
    (exp_dir / "run.json").write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return exp_dir


def update_experiment_index():
    """Call scripts/update_experiment_index.py to refresh index (best-effort)."""
    try:
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts" / "update_experiment_index.py")], check=True, cwd=ROOT)
        print("✅ Updated experiment index")
    except subprocess.CalledProcessError:
        print("⚠️ Failed to update experiment index, run manually: python scripts/update_experiment_index.py")


def main():
    parser = argparse.ArgumentParser(
        description="Reserve a new experiment ID and create directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/reserve_experiment.py --system G --title "tfidf_with_char_ngrams"
  python scripts/reserve_experiment.py --system D --title "roberta_base" --template "tfidf_baseline"
"""
    )

    parser.add_argument("--system", choices=VALID_SYSTEMS, required=True, help="Experiment system: G=Linear, D=Deep, E=Ensemble, T=Test")
    parser.add_argument("--title", required=True, help="Short descriptive title for the experiment")
    parser.add_argument("--template", help="Base configuration template to copy from (e.g., 'tfidf_baseline')")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be created without actually creating it")
    args = parser.parse_args()

    exp_id = find_next_experiment_id(args.system)
    full_title = f"{exp_id}_{args.title}"

    print(f"🔍 Next available experiment ID: {exp_id}")
    print(f"📝 Full title: {full_title}")
    print(f"🎯 System: {args.system}")

    if args.dry_run:
        print("\n🔍 DRY RUN - Would create:")
        print(f"  📁 experiments/{exp_id}/")
        print(f"  📄 config/experiments/{exp_id}.yaml")
        print(f"  🎯 artifacts/models/{exp_id}/ (during training)")
        return

    try:
        # Create experiment structure
        exp_dir = create_experiment_structure(exp_id, full_title, args.system, args.template)
        print(f"✅ Created experiment directory: {exp_dir}")

        # Create configuration
        config_path = create_config_template(exp_id, args.system, args.template)
        print(f"✅ Created configuration: {config_path}")

        # Ensure artifacts directory exists
        artifact_dir = ARTIFACTS_DIR / exp_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created artifacts directory: {artifact_dir}")

        # Persist assigned number to avoid future gaps/reuse
        try:
            num = int(exp_id[3:6])
            _write_last_assigned(num)
            print(f"🔒 Persisted last assigned experiment number: {num}")
        except Exception:
            pass

        # Update experiment index (best effort)
        update_experiment_index()

        print(f"\n🎉 Experiment {exp_id} reserved successfully!")
        print("\n📋 Next steps:")
        print(f"1. Edit configuration: config/experiments/{exp_id}.yaml")
        print(f"2. Update README: experiments/{exp_id}/README.md")
        print(f"3. Run training: python train.py --config config/experiments/{exp_id}.yaml")
        print(f"4. Check artifacts: artifacts/models/{exp_id}/")
    except Exception as e:
        print(f"❌ Error creating experiment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()