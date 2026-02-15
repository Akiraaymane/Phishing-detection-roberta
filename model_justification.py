"""
Documentation détaillée sur le choix du modèle RoBERTa pour la détection de phishing
Justification technique et scientifique
"""

# JUSTIFICATION DU CHOIX DU MODELE ROBERTA POUR LA DETECTION DE PHISHING

## 1. PROBLEMATIQUE
'''

La détection de phishing dans les SMS et emails est un problème de classification binaire
où le modèle doit distinguer entre:
- Classe 0: Messages légitimes (ham)
- Classe 1: Messages de phishing (spam)

Les défis spécifiques incluent:
- Textes courts et variables (SMS: 10-160 caractères en moyenne)
- Langage informel et abréviations
- Présence d'URLs, numéros de téléphone, symboles spéciaux
- Évolution constante des techniques de phishing
- Besoin de comprendre le contexte sémantique


## 2. CHOIX DU MODELE: RoBERTa (Robustly Optimized BERT Pretraining Approach)

### 2.1 Pourquoi un Transformer?

Les architectures Transformer présentent plusieurs avantages pour cette tâche:

1. ATTENTION MECHANISM
   - Capture les relations à longue distance dans le texte
   - Identifie les mots-clés importants (URGENT, FREE, WIN, CLICK)
   - Comprend le contexte même dans des textes courts

2. PRETRAINING
   - Pré-entraîné sur des milliards de tokens
   - Connaissance générale du langage naturel
   - Nécessite moins de données spécifiques au domaine

3. TRANSFER LEARNING
   - Fine-tuning rapide sur nos datasets
   - Convergence plus rapide que l'entraînement from scratch
   - Meilleures performances avec peu de données


### 2.2 Pourquoi RoBERTa spécifiquement?

RoBERTa améliore BERT sur plusieurs aspects critiques:

1. DYNAMIC MASKING
   - BERT: Masques statiques lors du preprocessing
   - RoBERTa: Masques différents à chaque epoch
   - Résultat: Meilleure généralisation

2. REMOVAL OF NSP (Next Sentence Prediction)
   - BERT utilise NSP qui n'est pas utile pour notre tâche
   - RoBERTa se concentre uniquement sur MLM (Masked Language Modeling)
   - Plus efficace pour la classification de texte

3. LARGER BATCH SIZES
   - Entraîné avec des batches plus grands
   - Stabilité d'entraînement améliorée
   - Meilleures performances finales

4. MORE TRAINING DATA
   - 160GB de texte (vs 16GB pour BERT)
   - Meilleure compréhension du langage
   - Vocabulaire plus riche


## 3. COMPARAISON AVEC D'AUTRES APPROCHES

### 3.1 vs Approches Traditionnelles

NAIVE BAYES / SVM:
- Avantages: Rapides, simples, interprétables
- Inconvénients: 
  * Ne capturent pas le contexte sémantique
  * Nécessitent feature engineering manuel
  * Performances limitées sur textes courts

LSTM/GRU:
- Avantages: Capturent les séquences, mémoire à long terme
- Inconvénients:
  * Training plus lent (séquentiel)
  * Difficulté avec les dépendances lointaines
  * Moins performants que les Transformers sur NLP

### 3.2 vs Autres Transformers

GPT-2/GPT-3:
- Optimisés pour la génération, pas la classification
- Plus gourmands en ressources
- Overkill pour notre tâche

BERT-base:
- Moins optimisé que RoBERTa
- Performances légèrement inférieures
- NSP inutile pour notre cas

DistilBERT:
- Plus rapide et léger
- Mais ~3% moins performant
- Acceptable si contraintes de ressources fortes

ALBERT:
- Partage de paramètres
- Plus lent à l'inférence malgré moins de paramètres


## 4. ARCHITECTURE TECHNIQUE

### 4.1 Architecture de RoBERTa-base

```
Input: Tokens (max 512)
    |
Embedding Layer (768 dim)
    |
12 Transformer Blocks:
    - Multi-Head Self-Attention (12 heads)
    - Feed-Forward Network (3072 hidden units)
    - Layer Normalization
    - Residual Connections
    |
Pooler (CLS token)
    |
Classification Head (2 classes)
    |
Output: [P(legitimate), P(phishing)]
```

Paramètres totaux: ~125M
Taille du modèle: ~500MB


### 4.2 Adaptations pour notre tâche

1. TRONCATION A 128 TOKENS
   - SMS/Emails rarement > 128 tokens
   - Réduit le temps de calcul de ~75%
   - Performances similaires

2. FINE-TUNING COMPLET
   - Tous les layers sont ajustés
   - Learning rate faible (2e-5)
   - Évite le catastrophic forgetting

3. EARLY STOPPING
   - Patience de 2 epochs
   - Évite l'overfitting
   - Optimise le temps d'entraînement


## 5. PERFORMANCE ATTENDUE

Basé sur la littérature et nos expériences:

BENCHMARKS SMS SPAM:
- Naive Bayes: ~95% accuracy
- SVM: ~96% accuracy
- LSTM: ~97% accuracy
- RoBERTa: ~98-99% accuracy

BENCHMARKS EMAIL SPAM:
- Règles traditionnelles: ~90% accuracy
- ML classique: ~94-96% accuracy
- Deep Learning: ~97-98% accuracy
- Transformers: ~98-99% accuracy


## 6. LIMITATIONS ET CONSIDERATIONS

### 6.1 Limitations

1. RESSOURCES COMPUTATIONNELLES
   - Nécessite 8GB+ RAM
   - Entraînement lent sur CPU (~1h pour SMS)
   - GPU recommandé pour production

2. TAILLE DU MODELE
   - 500MB par modèle
   - Temps de chargement non négligeable
   - Considérations pour déploiement mobile

3. INTERPRETABILITE
   - Modèle "boîte noire"
   - Difficile d'expliquer chaque prédiction
   - Nécessite attention visualization pour insights

### 6.2 Alternatives selon les contraintes

SI RESSOURCES LIMITEES:
- DistilRoBERTa (66% de taille, ~97% des performances)
- TinyBERT (très petit, ~95% des performances)

SI LATENCE CRITIQUE:
- FastText avec embeddings
- Distilled model avec quantization

SI INTERPRETABILITE REQUISE:
- Règles + ML traditionnel
- LIME/SHAP sur RoBERTa


## 7. CONCLUSION

RoBERTa est le choix optimal pour ce projet car:

1. PERFORMANCES: State-of-the-art sur NLP tasks
2. ROBUSTESSE: Pré-entraîné sur données massives
3. TRANSFERABILITE: Fine-tuning efficace avec peu de données
4. MATURITE: Modèle prouvé avec large communauté
5. DISPONIBILITE: HuggingFace facilite l'utilisation

Le surcoût en ressources est justifié par le gain significatif en précision,
particulièrement critique pour la détection de phishing où les faux négatifs
peuvent avoir des conséquences sérieuses.


## 8. REFERENCES

- Liu et al. (2019): "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
- Devlin et al. (2018): "BERT: Pre-training of Deep Bidirectional Transformers"
- Vaswani et al. (2017): "Attention Is All You Need"
- Raschka et al. (2020): "Machine Learning for Text Classification"


Auteur: Data Science Team
Date: 2026
Version: 1.0
"""

def get_model_info():
    """Retourne les informations sur le modèle sous forme de dictionnaire"""
    return {
        "model_name": "RoBERTa-base",
        "architecture": "Transformer",
        "parameters": 125_000_000,
        "model_size_mb": 500,
        "max_sequence_length": 128,
        "embedding_dimension": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "hidden_size": 3072,
        "training_approach": "Fine-tuning",
        "pretrained_on": "160GB of text data",
        "advantages": [
            "State-of-the-art performance",
            "Strong semantic understanding",
            "Robust to variations",
            "Pre-trained knowledge",
            "Easy fine-tuning"
        ],
        "limitations": [
            "High computational cost",
            "Large model size",
            "Slower inference than traditional ML",
            "Less interpretable"
        ]
    }


if __name__ == "__main__":
    import json
    
    print("Model Information:")
    print(json.dumps(get_model_info(), indent=2))

    '''