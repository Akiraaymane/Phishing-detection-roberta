"""
Script de validation croisée (K-Fold Cross-Validation)
Évalue la robustesse du modèle avec différents splits de données
"""

import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import os
from preprocessing import TextPreprocessor, load_sms_data, load_email_data


def compute_metrics(pred):
    """Calcule les métriques d'évaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class CrossValidator:
    """Classe pour effectuer la validation croisée"""
    
    def __init__(self, data_type, n_splits=5, epochs=2, batch_size=16):
        self.data_type = data_type
        self.n_splits = n_splits
        self.epochs = epochs
        self.batch_size = batch_size
        self.preprocessor = TextPreprocessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
    
    def run(self, df):
        """Exécute la validation croisée"""
        print(f"\nStarting {self.n_splits}-Fold Cross-Validation for {self.data_type.upper()}")
        print(f"Total samples: {len(df)}")
        print(f"Device: {self.device}\n")
        
        # Prétraitement
        df = self.preprocessor.preprocess(df)
        
        # Préparer les données
        X = df['text_clean'].values
        y = df['label'].values
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nFold {fold}/{self.n_splits}")
            print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")
            
            # Créer les datasets pour ce fold
            train_data = pd.DataFrame({
                'text_clean': X[train_idx],
                'label': y[train_idx]
            })
            val_data = pd.DataFrame({
                'text_clean': X[val_idx],
                'label': y[val_idx]
            })
            
            # Entraîner et évaluer
            metrics = self._train_and_evaluate(train_data, val_data, fold)
            fold_results.append(metrics)
            
            print(f"Fold {fold} Results:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
        
        # Calculer les statistiques globales
        self._calculate_statistics(fold_results)
        
        return fold_results
    
    def _train_and_evaluate(self, train_df, val_df, fold):
        """Entraîne et évalue le modèle pour un fold"""
        
        # Convertir en datasets HuggingFace
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        # Charger le tokenizer et le modèle
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
        model.to(self.device)
        
        # Fonction de tokenization
        def tokenize_function(examples):
            return tokenizer(
                examples['text_clean'],
                padding='max_length',
                truncation=True,
                max_length=128
            )
        
        # Tokenizer les datasets
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        # Renommer la colonne label
        train_dataset = train_dataset.rename_column('label', 'labels')
        val_dataset = val_dataset.rename_column('label', 'labels')
        
        # Définir le format PyTorch
        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        # Arguments d'entraînement
        training_args = TrainingArguments(
            output_dir=f'./cv_results_temp/fold_{fold}',
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=2e-5,
            warmup_steps=50,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="no",
            report_to="none",
        )
        
        # Créer le Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        
        # Entraîner
        trainer.train()
        
        # Évaluer
        eval_results = trainer.evaluate()
        
        # Nettoyer la mémoire
        del model
        del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            'fold': fold,
            'accuracy': eval_results['eval_accuracy'],
            'f1': eval_results['eval_f1'],
            'precision': eval_results['eval_precision'],
            'recall': eval_results['eval_recall']
        }
    
    def _calculate_statistics(self, fold_results):
        """Calcule les statistiques sur tous les folds"""
        
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        
        print(f"\n\nCross-Validation Summary ({self.n_splits} folds):")
        print("-" * 60)
        
        summary = {}
        
        for metric in metrics:
            values = [r[metric] for r in fold_results]
            mean = np.mean(values)
            std = np.std(values)
            
            summary[metric] = {
                'mean': float(mean),
                'std': float(std),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
            
            print(f"{metric.capitalize():12s}: {mean:.4f} (+/- {std:.4f})")
        
        # Sauvegarder les résultats
        output_dir = f'cv_results_{self.data_type}'
        os.makedirs(output_dir, exist_ok=True)
        
        results_file = f'{output_dir}/cross_validation_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'data_type': self.data_type,
                'n_splits': self.n_splits,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'fold_results': fold_results,
                'summary': summary
            }, f, indent=4)
        
        print(f"\nResults saved to {results_file}")
        
        # Créer un graphique
        self._plot_results(fold_results, summary)
    
    def _plot_results(self, fold_results, summary):
        """Génère un graphique des résultats de cross-validation"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        output_dir = f'cv_results_{self.data_type}'
        
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        folds = [r['fold'] for r in fold_results]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            values = [r[metric] for r in fold_results]
            mean = summary[metric]['mean']
            std = summary[metric]['std']
            
            axes[idx].plot(folds, values, 'o-', linewidth=2, markersize=8, label='Fold results')
            axes[idx].axhline(y=mean, color='r', linestyle='--', label=f'Mean: {mean:.4f}')
            axes[idx].fill_between(folds, mean-std, mean+std, alpha=0.2, color='r')
            
            axes[idx].set_xlabel('Fold')
            axes[idx].set_ylabel('Score')
            axes[idx].set_title(f'{metric.capitalize()} across Folds')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
            axes[idx].set_ylim([0, 1])
        
        plt.suptitle(f'{self.n_splits}-Fold Cross-Validation Results - {self.data_type.upper()}', 
                    fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/cross_validation_plot.png', dpi=300)
        plt.close()
        
        print(f"Plot saved to {output_dir}/cross_validation_plot.png")


def main():
    parser = argparse.ArgumentParser(description='K-Fold Cross-Validation for phishing detection')
    parser.add_argument('--data_type', type=str, required=True, choices=['sms', 'email'],
                       help='Type of data: sms or email')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to CSV dataset')
    parser.add_argument('--n_splits', type=int, default=5,
                       help='Number of folds (default: 5)')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of epochs per fold (default: 2)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    args = parser.parse_args()
    
    # Charger les données
    print(f"Loading {args.data_type} data from {args.data_path}...")
    if args.data_type == 'sms':
        df = load_sms_data(args.data_path)
    else:
        df = load_email_data(args.data_path)
    
    print(f"Loaded {len(df)} samples")
    
    # Créer le validateur
    validator = CrossValidator(
        data_type=args.data_type,
        n_splits=args.n_splits,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Exécuter la validation croisée
    validator.run(df)
    
    print("\nCross-validation complete!")


if __name__ == '__main__':
    main()
