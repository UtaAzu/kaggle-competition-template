"""
TF-IDF baseline training pipeline for Jigsaw competition.
"""
import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from ..utils.data import load_data, create_text_features, create_cv_splits, clean_data
from ..utils.config import (
    ensure_dirs, 
    get_experiment_dir, 
    validate_experiment_reservation,
    update_experiment_status
)

logger = logging.getLogger(__name__)


class TFIDFPipeline:
    """TF-IDF baseline pipeline for text classification."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vectorizer = None
        self.model = None
        self.cv_scores = []
        self.oof_predictions = None
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for training."""
        text_fields = self.config['features']['text_fields']
        df = create_text_features(df, text_fields)
        return df
    
    def setup_vectorizer(self) -> TfidfVectorizer:
        """Setup TF-IDF vectorizer."""
        tfidf_config = self.config['features']['tfidf']
        
        vectorizer = TfidfVectorizer(
            max_features=tfidf_config.get('max_features', 10000),
            ngram_range=tuple(tfidf_config.get('ngram_range', [1, 2])),
            min_df=tfidf_config.get('min_df', 2),
            max_df=tfidf_config.get('max_df', 0.95),
            stop_words=tfidf_config.get('stop_words', 'english'),
            sublinear_tf=tfidf_config.get('sublinear_tf', True)
        )
        
        return vectorizer
    
    def setup_model(self) -> LogisticRegression:
        """Setup classification model."""
        model_config = self.config['model']
        model_type = model_config.get('type', 'logistic_regression')
        
        if model_type == 'logistic_regression':
            params = model_config.get('params', {})
            model = LogisticRegression(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model
    
    def train_fold(self, 
                   X_train: np.ndarray, 
                   y_train: np.ndarray,
                   X_val: np.ndarray, 
                   y_val: np.ndarray) -> Tuple[float, np.ndarray]:
        """Train model on one fold."""
        # Clone model for this fold
        model = self.setup_model()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict validation set
        val_preds = model.predict_proba(X_val)[:, 1]
        
        # Calculate AUC
        auc_score = roc_auc_score(y_val, val_preds)
        
        return auc_score, val_preds
    
    def run_cross_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run cross-validation training."""
        logger.info("Starting cross-validation training...")
        
        # Clean data
        df = clean_data(df)
        
        # Create features
        df = self.create_features(df)
        
        # Setup vectorizer and fit on full data
        self.vectorizer = self.setup_vectorizer()
        X = self.vectorizer.fit_transform(df['combined_text'])
        
        target_col = self.config['training']['target_column']
        y = df[target_col].values
        
        # Create CV splits
        cv_splits = create_cv_splits(df, target_col, self.config['cv'])
        
        # Initialize OOF predictions
        oof_preds = np.zeros(len(df))
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            logger.info(f"Training fold {fold_idx + 1}/{len(cv_splits)}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train fold
            fold_score, val_preds = self.train_fold(X_train, y_train, X_val, y_val)
            
            # Store results
            oof_preds[val_idx] = val_preds
            fold_scores.append(fold_score)
            
            logger.info(f"Fold {fold_idx + 1} AUC: {fold_score:.4f}")
        
        # Calculate overall CV score
        cv_score = roc_auc_score(y, oof_preds)
        
        # Store results
        self.cv_scores = fold_scores
        self.oof_predictions = oof_preds
        
        results = {
            'cv_score': cv_score,
            'fold_scores': fold_scores,
            'cv_std': np.std(fold_scores),
            'n_folds': len(fold_scores)
        }
        
        logger.info(f"CV AUC: {cv_score:.4f} ± {np.std(fold_scores):.4f}")
        
        return results
    
    def train_final_model(self, df: pd.DataFrame) -> None:
        """Train final model on full dataset."""
        logger.info("Training final model on full dataset...")
        
        # Clean data
        df = clean_data(df)
        
        # Create features (vectorizer should already be fitted)
        df = self.create_features(df)
        X = self.vectorizer.transform(df['combined_text'])
        
        target_col = self.config['training']['target_column']
        y = df[target_col].values
        
        # Train final model
        self.model = self.setup_model()
        self.model.fit(X, y)
        
        logger.info("Final model training completed")
    
    def save_artifacts(self, results: Dict[str, Any]) -> None:
        """Save model artifacts and results."""
        logger.info("Saving artifacts...")
        
        # Ensure directories exist
        ensure_dirs(self.config)
        exp_dir = get_experiment_dir(self.config)
        
        # Save vectorizer
        vectorizer_path = exp_dir / "tfidf_vectorizer.pkl"
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Save model
        model_path = exp_dir / "model.pkl"
        joblib.dump(self.model, model_path)
        
        # Save OOF predictions
        oof_path = exp_dir / "oof_predictions.npy"
        np.save(oof_path, self.oof_predictions)
        
        # Save results summary
        results_path = exp_dir / "cv_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save config
        config_path = exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Artifacts saved to {exp_dir}")
        
        # Print summary
        print(f"\n=== Training Summary ===")
        print(f"CV AUC: {results['cv_score']:.4f} ± {results['cv_std']:.4f}")
        print(f"Fold scores: {[f'{score:.4f}' for score in results['fold_scores']]}")
        print(f"Artifacts saved to: {exp_dir}")


def run_tfidf_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run TF-IDF training pipeline."""
    # Validate experiment reservation first
    try:
        validate_experiment_reservation(config)
        logger.info("✅ Experiment reservation validated")
    except ValueError as e:
        logger.error(f"❌ Experiment validation failed: {e}")
        raise
    
    # Update experiment status to "running"
    update_experiment_status(config, "running")
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.get('logging', {}).get('level', 'INFO')),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load data
    data_config = config['data']
    df = load_data(data_config['train_path'], data_config.get('encoding', 'utf-8'))
    
    # Initialize pipeline
    pipeline = TFIDFPipeline(config)
    
    try:
        # Run cross-validation
        results = pipeline.run_cross_validation(df)
        
        # Train final model
        pipeline.train_final_model(df)
        
        # Save artifacts
        pipeline.save_artifacts(results)
        
        # Update experiment status to "completed" with results
        update_experiment_status(
            config, 
            "completed",
            cv=results,
            artifacts_saved=True
        )
        
        logger.info("✅ Training completed successfully and metadata updated")
        return results
        
    except Exception as e:
        # Update experiment status to "failed"
        update_experiment_status(
            config, 
            "failed",
            error=str(e)
        )
        logger.error(f"❌ Training failed: {e}")
        raise
    
    return results