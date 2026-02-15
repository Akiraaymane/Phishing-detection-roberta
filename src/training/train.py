import os
import time
import json
import argparse
import torch
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.preprocessing import TextPreprocessor, load_sms_data, load_email_data, split_data

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

def train_model(data_type, data_path, output_dir, epochs=2, batch_size=16, learning_rate=2e-5):
    """Entraîne le modèle RoBERTa"""
    
    # Charger les données
    print(f"Loading {data_type} data from {data_path}")
    if data_type == 'sms':
        df = load_sms_data(data_path)
    else:
        df = load_email_data(data_path)
    print(f"Loaded {len(df)} samples")
    
    # Prétraitement
    print("Preprocessing...")
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess(df)
    print(f"After preprocessing: {len(df)} samples")
    
    # Split des données
    train_df, val_df, test_df = split_data(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Convertir en Dataset HuggingFace
    train_dataset = Dataset.from_pandas(train_df[['text_clean', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text_clean', 'label']])
    
    # Charger tokenizer et modèle
    print("Loading RoBERTa model...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    
    # Fonction de tokenization
    def tokenize_function(examples):
        return tokenizer(
            examples['text_clean'], 
            padding='max_length', 
            truncation=True, 
            max_length=128
        )
    
    # Tokenizer les datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Renommer la colonne label en labels (requis par Trainer)
    train_dataset = train_dataset.rename_column('label', 'labels')
    val_dataset = val_dataset.rename_column('label', 'labels')
    
    # Définir le format PyTorch
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Arguments d'entraînement
    training_args = TrainingArguments(
        output_dir='./results_temp',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
    )
    
    # Créer le Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Entraîner
    print("Starting training...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    # Évaluer
    print("Evaluating...")
    eval_results = trainer.evaluate()
    
    # Sauvegarder le modèle
    print(f"Saving model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return {
        'training_time': training_time,
        'metrics': eval_results
    }

def main():
    parser = argparse.ArgumentParser(description='Train phishing detection model')
    parser.add_argument('--data_type', type=str, required=True, choices=['sms', 'email'],
                        help='Type of data: sms or email')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to CSV dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs (default: 2)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate (default: 2e-5)')
    args = parser.parse_args()
    
    # Détecter l'environnement
    env_type = os.environ.get('DOCKER_ENV', 'local')
    
    print("\n" + "="*60)
    print(f"ENVIRONMENT: {env_type.upper()}")
    print(f"DATA TYPE: {args.data_type.upper()}")
    print(f"EPOCHS: {args.epochs}")
    print(f"BATCH SIZE: {args.batch_size}")
    print("="*60 + "\n")
    
    # Entraîner le modèle
    results = train_model(
        data_type=args.data_type,
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Préparer les métriques
    metrics = {
        'environment': env_type,
        'data_type': args.data_type,
        'model': 'RoBERTa-Transformer',
        'architecture': 'Transformer (Deep Learning)',
        'training_time_seconds': round(results['training_time'], 2),
        'training_time_minutes': round(results['training_time'] / 60, 2),
        'accuracy': round(results['metrics']['eval_accuracy'], 4),
        'f1_score': round(results['metrics']['eval_f1'], 4),
        'precision': round(results['metrics']['eval_precision'], 4),
        'recall': round(results['metrics']['eval_recall'], 4),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    }
    
    # Sauvegarder les métriques en JSON
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    filename_json = os.path.join(results_dir, f"{env_type}_{args.data_type}_metrics.json")
    with open(filename_json, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {filename_json}")
    
    # Sauvegarder les métriques en TXT
    filename_txt = os.path.join(results_dir, f"{env_type}_{args.data_type}_metrics.txt")
    with open(filename_txt, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"PHISHING DETECTION TRAINING RESULTS - {env_type.upper()}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Environment: {env_type.upper()}\n")
        f.write(f"Data Type: {args.data_type.upper()}\n")
        f.write(f"Model: RoBERTa (Transformer - Deep Learning)\n\n")
        f.write(f"Training Configuration:\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  Learning Rate: {args.learning_rate}\n\n")
        f.write(f"Training Time:\n")
        f.write(f"  Seconds: {metrics['training_time_seconds']}s\n")
        f.write(f"  Minutes: {metrics['training_time_minutes']:.2f} min\n\n")
        f.write(f"Performance Metrics:\n")
        f.write(f"  Accuracy: {metrics['accuracy']*100:.2f}%\n")
        f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall: {metrics['recall']:.4f}\n\n")
        f.write("="*60 + "\n")
    print(f"Metrics saved to {filename_txt}")
    
    # Afficher le résumé
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Environment: {env_type.upper()}")
    print(f"Training Time: {metrics['training_time_minutes']:.2f} minutes")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
