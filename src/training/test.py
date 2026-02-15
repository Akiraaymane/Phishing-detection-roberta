"""
Script de test complet du modèle de détection de phishing
Génère des métriques détaillées, matrices de confusion, courbes ROC, etc.
"""

import os
import argparse
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from src.preprocessing import TextPreprocessor, load_sms_data, load_email_data, split_data


class ModelTester:
    """Classe pour tester et évaluer le modèle"""
    
    def __init__(self, model_path, data_type):
        self.model_path = model_path
        self.data_type = data_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = TextPreprocessor()
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def predict_batch(self, texts):
        """Prédiction par batch"""
        predictions = []
        probabilities = []
        
        for text in texts:
            cleaned_text = self.preprocessor.clean_text(text)
            inputs = self.tokenizer(
                cleaned_text, 
                return_tensors='pt', 
                truncation=True, 
                max_length=128, 
                padding='max_length'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs).item()
                prob = probs[0][pred].item()
                
                predictions.append(pred)
                probabilities.append(probs[0].cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)
    
    def evaluate(self, test_df):
        """Évaluation complète du modèle"""
        print(f"\n{'='*60}")
        print(f"TESTING MODEL ON {self.data_type.upper()} DATA")
        print(f"{'='*60}\n")
        
        print(f"Test set size: {len(test_df)}")
        
        # Prédictions
        start_time = time.time()
        predictions, probabilities = self.predict_batch(test_df['text_clean'].tolist())
        inference_time = time.time() - start_time
        
        # Labels réels
        true_labels = test_df['label'].values
        
        # Calcul des métriques
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )
        
        # Matrice de confusion
        cm = confusion_matrix(true_labels, predictions)
        
        # Classification report
        report = classification_report(
            true_labels, 
            predictions, 
            target_names=['Legitimate', 'Phishing'],
            output_dict=True
        )
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(true_labels, probabilities[:, 1])
            fpr, tpr, thresholds = roc_curve(true_labels, probabilities[:, 1])
        except Exception as e:
            print(f"Warning: Could not compute ROC curve: {e}")
            roc_auc = None
            fpr, tpr = None, None
        
        # Résultats
        results = {
            'data_type': self.data_type,
            'test_samples': len(test_df),
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(roc_auc, 4) if roc_auc else None,
            'confusion_matrix': cm.tolist(),
            'inference_time_seconds': round(inference_time, 2),
            'avg_time_per_sample_ms': round((inference_time / len(test_df)) * 1000, 2),
            'classification_report': report,
            'timestamp': datetime.now().isoformat()
        }
        
        # Affichage des résultats
        self._print_results(results, cm)
        
        # Génération des graphiques
        self._generate_plots(cm, fpr, tpr, roc_auc, results)
        
        return results, predictions, probabilities
    
    def _print_results(self, results, cm):
        """Affiche les résultats de manière formatée"""
        print(f"\n{'='*60}")
        print("TEST RESULTS")
        print(f"{'='*60}\n")
        
        print(f"Dataset: {results['data_type'].upper()}")
        print(f"Test Samples: {results['test_samples']}")
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:  {results['accuracy']*100:.2f}%")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1_score']:.4f}")
        if results['roc_auc']:
            print(f"  ROC AUC:   {results['roc_auc']:.4f}")
        
        print(f"\nInference Performance:")
        print(f"  Total Time: {results['inference_time_seconds']:.2f}s")
        print(f"  Avg Time per Sample: {results['avg_time_per_sample_ms']:.2f}ms")
        
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Legit  Phishing")
        print(f"Actual Legit     {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"       Phishing  {cm[1][0]:5d}  {cm[1][1]:5d}")
        
        print(f"\n{'='*60}\n")
    
    def _generate_plots(self, cm, fpr, tpr, roc_auc, results):
        """Génère les visualisations"""
        output_dir = os.path.join('results', f'test_results_{self.data_type}')
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Matrice de confusion
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legitimate', 'Phishing'],
                   yticklabels=['Legitimate', 'Phishing'])
        plt.title(f'Confusion Matrix - {self.data_type.upper()}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300)
        plt.close()
        
        # 2. Courbe ROC
        if fpr is not None and tpr is not None:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {self.data_type.upper()}')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/roc_curve.png', dpi=300)
            plt.close()
        
        # 3. Graphique des métriques
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [results['accuracy'], results['precision'], 
                 results['recall'], results['f1_score']]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
        plt.ylim([0, 1])
        plt.ylabel('Score')
        plt.title(f'Performance Metrics - {self.data_type.upper()}')
        plt.grid(axis='y', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value*100:.2f}%' if 'Accuracy' in metrics else f'{value:.4f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/metrics_comparison.png', dpi=300)
        plt.close()
        
        print(f"Plots saved to {output_dir}/")
    
    def analyze_errors(self, test_df, predictions):
        """Analyse des erreurs de prédiction"""
        print(f"\n{'='*60}")
        print("ERROR ANALYSIS")
        print(f"{'='*60}\n")
        
        # Identifier les erreurs
        errors = test_df[test_df['label'].values != predictions].copy()
        
        if len(errors) == 0:
            print("Perfect predictions! No errors found.")
            return
        
        errors['prediction'] = predictions[test_df['label'].values != predictions]
        
        # Faux positifs (prédit phishing, mais légitime)
        false_positives = errors[errors['label'] == 0]
        # Faux négatifs (prédit légitime, mais phishing)
        false_negatives = errors[errors['label'] == 1]
        
        print(f"Total Errors: {len(errors)} ({len(errors)/len(test_df)*100:.2f}%)")
        print(f"False Positives: {len(false_positives)} (Legitimate predicted as Phishing)")
        print(f"False Negatives: {len(false_negatives)} (Phishing predicted as Legitimate)")
        
        # Sauvegarder les erreurs
        output_dir = os.path.join('results', f'test_results_{self.data_type}')
        errors_file = f'{output_dir}/errors_analysis.csv'
        errors[['text_clean', 'label', 'prediction']].to_csv(errors_file, index=False)
        print(f"\nErrors saved to {errors_file}")
        
        # Afficher quelques exemples
        if len(false_positives) > 0:
            print(f"\n--- Sample False Positives (Top 3) ---")
            for i, row in false_positives.head(3).iterrows():
                print(f"{i+1}. {row['text_clean'][:100]}...")
        
        if len(false_negatives) > 0:
            print(f"\n--- Sample False Negatives (Top 3) ---")
            for i, row in false_negatives.head(3).iterrows():
                print(f"{i+1}. {row['text_clean'][:100]}...")
        
        print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Test phishing detection model')
    parser.add_argument('--data_type', type=str, required=True, choices=['sms', 'email'],
                       help='Type of data: sms or email')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to CSV dataset')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model directory')
    args = parser.parse_args()
    
    # Charger les données
    print(f"Loading {args.data_type} data...")
    if args.data_type == 'sms':
        df = load_sms_data(args.data_path)
    else:
        df = load_email_data(args.data_path)
    
    # Prétraitement
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess(df)
    
    # Split des données
    _, _, test_df = split_data(df)
    
    # Créer le testeur
    tester = ModelTester(args.model_path, args.data_type)
    
    # Évaluation
    results, predictions, probabilities = tester.evaluate(test_df)
    
    # Analyse des erreurs
    tester.analyze_errors(test_df, predictions)
    
    # Sauvegarder les résultats
    output_dir = os.path.join('results', f'test_results_{args.data_type}')
    results_file = f'{output_dir}/test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_file}")
    
    print(f"\n{'='*60}")
    print("TESTING COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
