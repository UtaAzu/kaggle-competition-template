#!/usr/bin/env python3
"""
List Experiments Tool

Show all experiments with their status and basic information.

Usage:
    python scripts/list_experiments.py
    python scripts/list_experiments.py --status running
    python scripts/list_experiments.py --system G
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"


def load_experiment_metadata(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """Load experiment metadata from run.json."""
    run_json = exp_dir / "run.json"
    if not run_json.exists():
        return None
    
    try:
        with open(run_json, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def list_experiments(status_filter: Optional[str] = None, 
                    system_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all experiments with optional filters."""
    experiments = []
    
    if not EXPERIMENTS_DIR.exists():
        return experiments
    
    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith('_'):
            continue
        
        metadata = load_experiment_metadata(exp_dir)
        if not metadata:
            # Create basic metadata for experiments without run.json
            metadata = {
                "experiment_id": exp_dir.name,
                "title": exp_dir.name,
                "system": exp_dir.name[-1] if len(exp_dir.name) > 0 else "?",
                "status": "unknown",
                "created_at": None
            }
        
        # Apply filters
        if status_filter and metadata.get("status") != status_filter:
            continue
        
        if system_filter and metadata.get("system") != system_filter:
            continue
        
        # Add computed fields
        metadata["has_artifacts"] = check_artifacts_exist(exp_dir.name)
        metadata["config_exists"] = check_config_exists(exp_dir.name)
        
        experiments.append(metadata)
    
    return experiments


def check_artifacts_exist(exp_id: str) -> bool:
    """Check if artifacts directory exists and has files."""
    artifacts_dir = ROOT / "artifacts" / "models" / exp_id
    if not artifacts_dir.exists():
        return False
    
    # Check if directory has any .pkl or .json files
    for file in artifacts_dir.iterdir():
        if file.suffix in ['.pkl', '.json', '.npy']:
            return True
    
    return False


def check_config_exists(exp_id: str) -> bool:
    """Check if experiment configuration exists."""
    config_path = ROOT / "config" / "experiments" / f"{exp_id}.yaml"
    return config_path.exists()


def format_status(status: str) -> str:
    """Format status with colors/emoji."""
    status_map = {
        "reserved": "🔒 Reserved",
        "running": "🏃 Running", 
        "completed": "✅ Completed",
        "failed": "❌ Failed",
        "unknown": "❓ Unknown"
    }
    return status_map.get(status, f"❓ {status}")


def format_date(date_str: Optional[str]) -> str:
    """Format date string."""
    if not date_str:
        return "-"
    
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M')
    except (ValueError, AttributeError):
        return date_str[:16] if date_str else "-"


def print_experiments_table(experiments: List[Dict[str, Any]]) -> None:
    """Print experiments in a formatted table."""
    if not experiments:
        print("No experiments found.")
        return
    
    print("\n📊 Experiments Overview")
    print("=" * 80)
    
    # Header
    print(f"{'ID':<12} {'Status':<15} {'System':<8} {'Created':<18} {'Artifacts':<10} {'Config':<8}")
    print("-" * 80)
    
    # Rows
    for exp in experiments:
        exp_id = exp.get("experiment_id", "")[:11]
        status = format_status(exp.get("status", "unknown"))
        system = exp.get("system", "?")
        created = format_date(exp.get("created_at"))
        artifacts = "✅" if exp.get("has_artifacts") else "❌"
        config = "✅" if exp.get("config_exists") else "❌"
        
        print(f"{exp_id:<12} {status:<25} {system:<8} {created:<18} {artifacts:<10} {config:<8}")
    
    print("-" * 80)
    print(f"Total: {len(experiments)} experiments")


def print_experiments_detailed(experiments: List[Dict[str, Any]]) -> None:
    """Print detailed experiment information."""
    for exp in experiments:
        exp_id = exp.get("experiment_id", "Unknown")
        title = exp.get("title", "No title")
        status = exp.get("status", "unknown")
        system = exp.get("system", "?")
        created = format_date(exp.get("created_at"))
        
        print(f"\n🧪 {exp_id}")
        print(f"   Title: {title}")
        print(f"   Status: {format_status(status)}")
        print(f"   System: {system}")
        print(f"   Created: {created}")
        
        if "cv" in exp and exp["cv"]:
            cv = exp["cv"]
            if "cv_score" in cv:
                print(f"   CV Score: {cv['cv_score']:.4f}")
        
        artifacts_status = "✅ Present" if exp.get("has_artifacts") else "❌ Missing"
        config_status = "✅ Present" if exp.get("config_exists") else "❌ Missing"
        print(f"   Artifacts: {artifacts_status}")
        print(f"   Config: {config_status}")


def main():
    parser = argparse.ArgumentParser(
        description="List experiments with their status and information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/list_experiments.py                    # List all experiments
  python scripts/list_experiments.py --status completed # Only completed experiments
  python scripts/list_experiments.py --system G         # Only G-system experiments
  python scripts/list_experiments.py --detailed         # Detailed view
        """
    )
    
    parser.add_argument(
        "--status",
        choices=["reserved", "running", "completed", "failed"],
        help="Filter by experiment status"
    )
    
    parser.add_argument(
        "--system",
        choices=["G", "D", "E", "T"],
        help="Filter by experiment system"
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed information instead of table"
    )
    
    args = parser.parse_args()
    
    # List experiments
    experiments = list_experiments(args.status, args.system)
    
    if args.detailed:
        print_experiments_detailed(experiments)
    else:
        print_experiments_table(experiments)
    
    # Show summary statistics
    if experiments:
        status_counts = {}
        system_counts = {}
        
        for exp in experiments:
            status = exp.get("status", "unknown")
            system = exp.get("system", "?")
            
            status_counts[status] = status_counts.get(status, 0) + 1
            system_counts[system] = system_counts.get(system, 0) + 1
        
        print(f"\n📈 Summary:")
        print(f"   By Status: {dict(status_counts)}")
        print(f"   By System: {dict(system_counts)}")


if __name__ == "__main__":
    main()