# GUIDE DE DEMARRAGE RAPIDE - PHISHING DETECTION

## Installation en 5 minutes

### Prérequis
- Python 3.8+
- 8GB RAM minimum
- 10GB espace disque

### Étape 1: Installation

```bash
# Cloner et accéder au projet
cd phishing-detection

# Créer environnement virtuel
python -m venv venv

# Activer (Windows)
venv\Scripts\activate
# Activer (Linux/Mac)
source venv/bin/activate

# Installer
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Étape 2: Workflow Automatisé

```bash
# Exécuter TOUT le pipeline automatiquement
python run_workflow.py --data_type sms

# Options:
#   --skip_docker : Sauter Docker training
#   --skip_cv : Sauter cross-validation
#   --step eda : Exécuter seulement l'EDA
#   --step train : Exécuter seulement training
```

### Étape 3: Démarrage Manuel (si problème)

```bash
# 1. Analyse exploratoire
python eda_analysis.py --data_type sms --data_path sms_spam.csv

# 2. Entraînement
python train.py --data_type sms --data_path sms_spam.csv --output_dir roberta-sms --epochs 2 --batch_size 16

# 3. Test
python test.py --data_type sms --data_path sms_spam.csv --model_path roberta-sms

# 4. API
python app.py
```

### Étape 4: Avec Docker

```bash
# Training
docker-compose up train-sms

# API
docker-compose up api
```

## Commandes Utiles

### Tests
```bash
pytest test_api.py -v
```

### Rapport
```bash
python generate_report.py
```

### Comparaison
```bash
python compare_environments.py --data_type sms
```

## Fichiers Générés

Après exécution:
```
roberta-sms/                      # Modèle entraîné
local_sms_metrics.json            # Métriques local
test_results_sms/                 # Résultats tests
eda_results_sms/                  # Analyses
RAPPORT_PROJET_YYYYMMDD.txt       # Rapport
```

## Accès API

Une fois l'API lancée:
- Interface: http://localhost:8000
- Health: http://localhost:8000/health
- Predict: POST http://localhost:8000/predict

## Dépannage

### Erreur mémoire
```bash
python train.py --batch_size 8  # Réduire batch size
```

### Packages manquants
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Docker problème
```bash
docker-compose down -v
docker-compose build --no-cache
```

## Temps d'Exécution Estimés

- EDA: 2-3 minutes
- Training SMS: 60-70 minutes (CPU)
- Testing: 5-10 minutes
- Cross-validation (3-fold): 180 minutes
- Pipeline complet: ~2.5 heures

## Support

Consulter:
1. README_COMPLET.md : Documentation complète
2. GUIDE-INSTALLATION.txt : Installation détaillée
3. model_justification.py : Justification technique

## Workflow Complet

```
EDA → Training → Testing → Cross-Validation → Docker Training → 
Comparison → API Testing → Report Generation
```

Tout automatisé avec:
```bash
python run_workflow.py --data_type sms
```

