# Phishing Detection System - RoBERTa Transformer

Système de détection de phishing pour SMS et Email utilisant l'architecture Deep Learning RoBERTa (Transformer).

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture du projet](#architecture-du-projet)
3. [Prérequis](#prerequis)
4. [Installation](#installation)
5. [Utilisation](#utilisation)
6. [Tests](#tests)
7. [Docker](#docker)
8. [MLOps](#mlops)
9. [Résultats](#resultats)
10. [Documentation](#documentation)

## Vue d'ensemble

### Problématique

La détection de phishing dans les SMS et emails est cruciale pour la cybersécurité. Ce projet implémente un système basé sur le Deep Learning capable de classifier automatiquement les messages comme légitimes ou malveillants.

### Solution

Architecture Transformer (RoBERTa) fine-tunée sur des datasets de phishing SMS et Email avec:
- Accuracy: ~98%
- F1-Score: ~0.93
- Temps d'inférence: < 100ms par message

### Choix du modèle

RoBERTa (Robustly Optimized BERT Pretraining Approach) a été choisi car:
- State-of-the-art pour les tâches de NLP
- Pré-entraîné sur 160GB de texte
- Excellent pour la compréhension sémantique
- Fine-tuning efficace avec peu de données

Voir `model_justification.py` pour la justification détaillée.

## Architecture du projet

```
phishing-detection/
├── data/
│   ├── sms_spam.csv              # Dataset SMS (5.5K messages)
│   └── email_dataset.csv         # Dataset Email (~500K messages)
├── models/
│   ├── roberta-sms/              # Modèle entraîné pour SMS
│   └── roberta-email/            # Modèle entraîné pour Email
├── scripts/
│   ├── preprocessing.py          # Nettoyage et préparation des données
│   ├── train.py                  # Entraînement du modèle
│   ├── test.py                   # Évaluation complète
│   ├── eda_analysis.py           # Analyse exploratoire
│   ├── cross_validation.py       # Validation croisée K-Fold
│   ├── compare_environments.py   # Comparaison Local vs Docker
│   └── mlops_setup.py            # Configuration MLOps
├── api/
│   ├── app.py                    # API FastAPI
│   └── test_api.py               # Tests unitaires de l'API
├── docker/
│   ├── Dockerfile.train          # Container pour l'entraînement
│   ├── Dockerfile.api            # Container pour l'API
│   └── docker-compose.yml        # Orchestration
├── results/
│   ├── test_results_sms/         # Résultats des tests SMS
│   ├── test_results_email/       # Résultats des tests Email
│   ├── eda_results_sms/          # Analyses exploratoires SMS
│   ├── eda_results_email/        # Analyses exploratoires Email
│   └── comparison_results/       # Comparaisons environnements
├── mlops_config/                 # Configuration MLOps
├── experiments/                  # Tracking des expériences
├── model_versions/               # Versioning des modèles
├── requirements.txt              # Dépendances Python
└── README.md                     # Ce fichier
```

## Prérequis

### Matériel

- RAM: 8GB minimum (16GB recommandé)
- Disque: 10GB d'espace libre
- CPU: Multi-core recommandé
- GPU: Optionnel (accélère l'entraînement ~10x)

### Logiciels

- Python 3.8 ou supérieur
- Docker et Docker Compose (pour containerisation)
- Git (pour versioning)

## Installation

### Méthode 1: Installation locale

```bash
# Cloner le repository
git clone <repository_url>
cd phishing-detection

# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# Télécharger les données NLTK
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Méthode 2: Installation avec Docker

```bash
# Construire les images
docker-compose build

# Les containers sont prêts à être utilisés
```

## Utilisation

### 1. Analyse Exploratoire des Données (EDA)

```bash
# Analyse du dataset SMS
python eda_analysis.py --data_type sms --data_path sms_spam.csv

# Analyse du dataset Email
python eda_analysis.py --data_type email --data_path email_dataset.csv
```

Génère des visualisations dans `eda_results_{data_type}/`:
- Distribution des classes
- Analyse de la longueur des textes
- Word clouds
- N-grams
- Corrélations

### 2. Entraînement du modèle

#### Local

```bash
# Entraînement SMS
python train.py --data_type sms \
                --data_path sms_spam.csv \
                --output_dir roberta-sms \
                --epochs 2 \
                --batch_size 16

# Entraînement Email
python train.py --data_type email \
                --data_path email_dataset.csv \
                --output_dir roberta-email \
                --epochs 2 \
                --batch_size 16
```

#### Docker

```bash
# Entraîner les deux modèles avec Docker
docker-compose up train-sms train-email

# Les modèles entraînés seront dans ./roberta-sms/ et ./roberta-email/
```

### 3. Évaluation du modèle

```bash
# Test du modèle SMS
python test.py --data_type sms \
               --data_path sms_spam.csv \
               --model_path roberta-sms

# Génère:
# - Matrice de confusion
# - Courbe ROC
# - Métriques détaillées
# - Analyse des erreurs
```

### 4. Validation croisée

```bash
# K-Fold Cross-Validation (5 folds)
python cross_validation.py --data_type sms \
                           --data_path sms_spam.csv \
                           --n_splits 5 \
                           --epochs 2
```

### 5. Comparaison Local vs Docker

```bash
# Après avoir entraîné localement ET avec Docker:
python compare_environments.py --data_type all

# Génère un rapport comparatif avec graphiques
```

### 6. Lancer l'API

#### Local

```bash
python app.py
```

#### Docker

```bash
docker-compose up api
```

L'API est accessible à: `http://localhost:8000`

### 7. Utilisation de l'API

#### Interface Web

Ouvrir `http://localhost:8000` dans un navigateur.

#### Requête cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "URGENT! You have won $1000000. Click here to claim!",
    "data_type": "sms"
  }'
```

#### Python

```python
import requests

response = requests.post('http://localhost:8000/predict', json={
    'text': 'Your account has been compromised. Verify now!',
    'data_type': 'email'
})

print(response.json())
# {'is_phishing': True, 'confidence': 0.95, 'label': 'PHISHING', 'data_type': 'email'}
```

## Tests

### Tests unitaires de l'API

```bash
# Exécuter tous les tests
pytest test_api.py -v

# Tests avec coverage
pytest test_api.py --cov=app --cov-report=html
```

### Tests de performance

```bash
# Load testing (nécessite locust)
pip install locust
locust -f load_test.py --host=http://localhost:8000
```

## Docker

### Architecture Docker

Le projet utilise 3 containers:

1. **train-sms**: Entraînement du modèle SMS
2. **train-email**: Entraînement du modèle Email
3. **api**: API de prédiction

### Commandes Docker utiles

```bash
# Construire les images
docker-compose build

# Entraîner uniquement SMS
docker-compose up train-sms

# Entraîner uniquement Email
docker-compose up train-email

# Lancer l'API
docker-compose up api

# Tout arrêter
docker-compose down

# Nettoyer les volumes
docker-compose down -v

# Logs d'un service
docker-compose logs api
```

### Optimisations Docker

- Multi-stage builds pour réduire la taille
- Cache de dépendances
- Volumes pour persistance des données
- Health checks pour monitoring

## MLOps

### Configuration MLOps

```bash
# Initialiser la configuration MLOps
python mlops_setup.py
```

Crée:
- Configuration du projet
- Tracking des expériences
- Versioning des modèles
- Snapshots d'environnement

### Tracking des expériences

```python
from mlops_setup import ExperimentTracker

tracker = ExperimentTracker()

# Logger une expérience
experiment_id = tracker.log_experiment(
    experiment_name='sms_roberta_v1',
    config={'epochs': 2, 'batch_size': 16},
    metrics={'accuracy': 0.98, 'f1': 0.93}
)

# Trouver la meilleure expérience
best = tracker.get_best_experiment(metric='accuracy')
```

### Versioning des modèles

```python
from mlops_setup import ModelVersioning

versioning = ModelVersioning()

# Enregistrer un modèle
version = versioning.register_model(
    model_path='roberta-sms',
    data_type='sms',
    metrics={'accuracy': 0.98}
)

# Promouvoir en production
versioning.promote_to_production(version)
```

### Reproductibilité

Tous les seeds sont fixés:
- Python random: seed=42
- NumPy: seed=42
- PyTorch: seed=42

Git commit hash enregistré pour chaque expérience.

## Résultats

### Performance SMS

```
Accuracy:  98.28%
F1-Score:  0.9278
Precision: 0.9000
Recall:    0.9574
```

### Performance Email

```
Accuracy:  ~97-98%
F1-Score:  ~0.92-0.94
Precision: ~0.90-0.93
Recall:    ~0.94-0.96
```

### Temps d'exécution

| Opération | Local (CPU) | Docker (CPU) |
|-----------|------------|--------------|
| Training SMS | 65 min | 70 min |
| Training Email | 80 min | 85 min |
| Inference | < 100 ms | < 120 ms |

### Comparaison avec autres méthodes

| Modèle | Accuracy | F1-Score | Taille | Inference |
|--------|----------|----------|--------|-----------|
| Naive Bayes | 95% | 0.88 | <1 MB | 1 ms |
| SVM | 96% | 0.90 | <5 MB | 5 ms |
| LSTM | 97% | 0.92 | 50 MB | 20 ms |
| **RoBERTa** | **98%** | **0.93** | **500 MB** | **100 ms** |

## Documentation

### Fichiers de documentation

- `model_justification.py`: Justification détaillée du choix de RoBERTa
- `GUIDE-INSTALLATION.txt`: Guide d'installation pas à pas
- `mlops_setup.py`: Documentation de la configuration MLOps

### Structure des résultats

Tous les résultats sont organisés par type:

```
results/
├── test_results_sms/
│   ├── test_results.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── metrics_comparison.png
│   └── errors_analysis.csv
├── eda_results_sms/
│   ├── class_distribution.png
│   ├── text_length_analysis.png
│   ├── word_frequency.png
│   └── wordclouds.png
└── comparison_results/
    ├── comparison_sms.csv
    ├── performance_comparison_sms.png
    └── time_comparison_sms.png
```

## Contribution

### Structure de commit

```
type(scope): description

[optional body]

[optional footer]
```

Types: feat, fix, docs, style, refactor, test, chore

### Tests avant commit

```bash
# Linting
flake8 *.py

# Tests
pytest test_api.py -v

# Coverage
pytest --cov=. --cov-report=html
```

## Licence

Ce projet est sous licence MIT.

## Auteurs

Data Science Team - 2026

## Support

Pour toute question ou problème:
1. Consulter `GUIDE-INSTALLATION.txt`
2. Vérifier les issues GitHub
3. Contacter l'équipe

## Roadmap

### Version 1.1 (Q2 2026)
- [ ] Support multilingue
- [ ] Fine-tuning avec données spécifiques au domaine
- [ ] API GraphQL
- [ ] Dashboard de monitoring en temps réel

### Version 2.0 (Q4 2026)
- [ ] Modèle léger pour mobile (TFLite/ONNX)
- [ ] Active learning pour amélioration continue
- [ ] Détection d'anomalies
- [ ] Intégration avec SIEM
