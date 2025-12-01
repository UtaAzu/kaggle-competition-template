"""
Configuration management utilities for Kaggle competition.
"""
import yaml
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def ensure_dirs(config: Dict[str, Any]) -> None:
    """Ensure artifact directories exist."""
    artifacts = config.get('artifacts', {})
    
    for dir_key in ['models_dir', 'preprocessors_dir']:
        dir_path = artifacts.get(dir_key)
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_experiment_dir(config: Dict[str, Any]) -> Path:
    """Get experiment directory path."""
    artifacts = config.get('artifacts', {})
    models_dir = artifacts.get('models_dir', 'artifacts/models')
    exp_name = artifacts.get('experiment_name', 'default')
    
    exp_dir = Path(models_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir


def validate_experiment_reservation(config: Dict[str, Any]) -> bool:
    """Validate that the experiment has been properly reserved."""
    artifacts = config.get('artifacts', {})
    exp_name = artifacts.get('experiment_name')
    
    if not exp_name:
        raise ValueError("Missing experiment_name in config artifacts section")
    
    # Check if experiment directory exists
    exp_dir = Path("experiments") / exp_name
    if not exp_dir.exists():
        raise ValueError(
            f"Experiment {exp_name} not found in experiments/ directory. "
            f"Reserve it first with: python scripts/reserve_experiment.py"
        )
    
    # Check if run.json exists
    run_json_path = exp_dir / "run.json"
    if not run_json_path.exists():
        raise ValueError(
            f"Missing run.json for experiment {exp_name}. "
            f"Re-reserve with: python scripts/reserve_experiment.py"
        )
    
    return True


def update_experiment_status(config: Dict[str, Any], status: str, **updates) -> None:
    """Update experiment status and metadata."""
    artifacts = config.get('artifacts', {})
    exp_name = artifacts.get('experiment_name')
    
    if not exp_name:
        return
    
    run_json_path = Path("experiments") / exp_name / "run.json"
    if not run_json_path.exists():
        return
    
    # Load existing metadata
    with open(run_json_path, 'r') as f:
        metadata = json.load(f)
    
    # Update status and other fields
    metadata['status'] = status
    metadata['updated_at'] = datetime.now().isoformat()
    
    # Add git commit if available
    if 'git_commit' not in metadata or not metadata['git_commit']:
        metadata['git_commit'] = get_git_commit()
    
    # Update with any additional fields
    metadata.update(updates)
    
    # Save updated metadata
    with open(run_json_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_experiment_metadata(exp_name: str) -> Optional[Dict[str, Any]]:
    """Get experiment metadata from run.json."""
    run_json_path = Path("experiments") / exp_name / "run.json"
    if not run_json_path.exists():
        return None
    
    with open(run_json_path, 'r') as f:
        return json.load(f)