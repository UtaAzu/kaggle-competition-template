#!/usr/bin/env python3
"""
Main prediction entry point for Kaggle competition.
Usage: python predict.py --config config/tfidf_baseline.yaml --test test.csv --output submission.csv
"""
import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config import load_config
from src.pipelines.predict import run_prediction


def main():
    parser = argparse.ArgumentParser(description="Make predictions for Kaggle competition")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--test", 
        type=str, 
        required=True,
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="submission.csv",
        help="Output submission file path"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run prediction
    submission = run_prediction(config, args.test, args.output)
    
    print(f"Predictions completed successfully!")
    print(f"Submission saved to: {args.output}")
    print(f"Submission shape: {submission.shape}")
    
    return submission


if __name__ == "__main__":
    main()