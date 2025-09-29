"""
LLM fine-tuning pipeline for Jigsaw competition with Chain of Thought and Calibrated Outputs.
"""
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from ..utils.data import load_data, create_cv_splits, clean_data
from ..utils.config import (
    ensure_dirs,
    get_experiment_dir,
    validate_experiment_reservation,
    update_experiment_status
)

logger = logging.getLogger(__name__)


class LLMDataset(Dataset):
    """Dataset for LLM fine-tuning."""
    def __init__(self, texts, labels, tokenizer, max_length=1024):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class LLMPipeline:
    """LLM fine-tuning pipeline with CoT and calibrated outputs."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.cv_scores = []
        self.oof_predictions = None

    def create_prompt(self, row):
        """Create prompt with CoT and calibrated outputs."""
        template = self.config['model']['params']['prompt_template']
        prompt = template.format(
            rule=row['rule'],
            positive_example_1=row['positive_example_1'],
            negative_example_1=row['negative_example_1'],
            body=row['body']
        )
        return prompt

    def setup_tokenizer(self):
        """Setup tokenizer."""
        model_name = self.config['model']['params']['model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup_model(self):
        """Setup model with LoRA."""
        model_name = self.config['model']['params']['model_name']
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # LoRA config
        lora_config = LoraConfig(
            r=self.config['model']['params']['lora_r'],
            lora_alpha=self.config['model']['params']['lora_alpha'],
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)

    def train_fold(self, train_texts, train_labels, val_texts, val_labels):
        """Train model on one fold."""
        # Create datasets
        train_dataset = LLMDataset(train_texts, train_labels, self.tokenizer, self.config['model']['params']['max_length'])
        val_dataset = LLMDataset(val_texts, val_labels, self.tokenizer, self.config['model']['params']['max_length'])

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=self.config['model']['params']['epochs'],
            per_device_train_batch_size=self.config['model']['params']['batch_size'],
            per_device_eval_batch_size=self.config['model']['params']['batch_size'],
            learning_rate=self.config['model']['params']['lr'],
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_dir="./logs",
            logging_steps=10,
            save_total_limit=2,
            fp16=True,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
        )

        # Train
        trainer.train()

        # Predict (simplified: use loss as proxy for now)
        eval_results = trainer.evaluate()
        auc_score = 0.5  # Placeholder, need proper prediction logic

        return auc_score, np.zeros(len(val_labels))  # Placeholder

    def run_cross_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run cross-validation training."""
        logger.info("Starting LLM cross-validation training...")

        # Clean data
        df = clean_data(df)

        # Create prompts
        df['prompt'] = df.apply(self.create_prompt, axis=1)

        # CV splits
        cv_config = self.config['cv']
        if cv_config.get('loo_rule', False):
            # LOO-Rule implementation
            rules = df['rule'].unique()
            for rule in rules:
                train_df = df[df['rule'] != rule]
                val_df = df[df['rule'] == rule]
                # Train and validate
                auc, preds = self.train_fold(
                    train_df['prompt'].tolist(), train_df[self.config['training']['target_column']].tolist(),
                    val_df['prompt'].tolist(), val_df[self.config['training']['target_column']].tolist()
                )
                self.cv_scores.append(auc)
        else:
            # Standard CV
            splits = create_cv_splits(df, cv_config)
            for fold, (train_idx, val_idx) in enumerate(splits):
                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]
                auc, preds = self.train_fold(
                    train_df['prompt'].tolist(), train_df[self.config['training']['target_column']].tolist(),
                    val_df['prompt'].tolist(), val_df[self.config['training']['target_column']].tolist()
                )
                self.cv_scores.append(auc)

        # Calculate mean CV score
        mean_cv_score = np.mean(self.cv_scores)

        results = {
            'cv_scores': self.cv_scores,
            'mean_cv_score': mean_cv_score,
            'oof_predictions': self.oof_predictions
        }

        return results


def run_llm_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run LLM training pipeline."""
    # Validate experiment
    validate_experiment_reservation(config)

    # Setup directories
    ensure_dirs(config)

    # Load data
    df = load_data(config['data']['train_path'], config['data']['encoding'])

    # Initialize pipeline
    pipeline = LLMPipeline(config)

    # Setup tokenizer and model
    pipeline.setup_tokenizer()
    pipeline.setup_model()

    # Run CV
    results = pipeline.run_cross_validation(df)

    # Save results
    exp_dir = get_experiment_dir(config)
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f)

    # Update experiment status
    update_experiment_status(config, "completed")

    return results
