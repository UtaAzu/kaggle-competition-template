"""
Prediction pipeline for Kaggle competition.
"""
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, Any

from ..utils.data import load_data, create_text_features, clean_data
from ..utils.config import get_experiment_dir

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """Pipeline for making predictions with trained models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vectorizer = None
        self.model = None
        
    def load_artifacts(self) -> None:
        """Load trained model artifacts."""
        logger.info("Loading trained artifacts...")
        
        exp_dir = get_experiment_dir(self.config)
        
        # Load vectorizer
        vectorizer_path = exp_dir / "tfidf_vectorizer.pkl"
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")
        self.vectorizer = joblib.load(vectorizer_path)
        
        # Load model
        model_path = exp_dir / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = joblib.load(model_path)
        
        logger.info("Artifacts loaded successfully")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for prediction (same as training)."""
        text_fields = self.config['features']['text_fields']
        df = create_text_features(df, text_fields)
        return df
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on test data."""
        logger.info(f"Making predictions on {len(df)} samples...")
        
        # Clean data
        df = clean_data(df)
        
        # Create features
        df = self.create_features(df)
        
        # Transform text to features
        X = self.vectorizer.transform(df['combined_text'])
        
        # Make predictions (return probabilities)
        predictions = self.model.predict_proba(X)[:, 1]
        
        logger.info("Predictions completed")
        return predictions
    
    def create_submission(self, 
                         df: pd.DataFrame, 
                         predictions: np.ndarray,
                         output_path: str = "submission.csv") -> pd.DataFrame:
        """Create submission file."""
        submission = pd.DataFrame({
            'row_id': df['row_id'],
            'rule_violation': predictions
        })
        
        # Save submission
        submission.to_csv(output_path, index=False)
        logger.info(f"Submission saved to {output_path}")
        
        return submission


def run_prediction(config: Dict[str, Any], test_path: str, output_path: str = "submission.csv") -> pd.DataFrame:
    """Run prediction pipeline."""
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.get('logging', {}).get('level', 'INFO')),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load test data
    encoding = config['data'].get('encoding', 'utf-8')
    test_df = load_data(test_path, encoding)
    
    # Initialize pipeline
    pipeline = PredictionPipeline(config)
    
    # Load trained artifacts
    pipeline.load_artifacts()
    
    # Make predictions
    predictions = pipeline.predict(test_df)
    
    # Create submission
    submission = pipeline.create_submission(test_df, predictions, output_path)
    
    return submission