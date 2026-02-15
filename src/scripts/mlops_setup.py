"""
Configuration MLOps et gestion du cycle de vie du modèle
Reproductibilité, versioning, et suivi des expériences
"""

import os
import json
import hashlib
from datetime import datetime
import subprocess


class MLOpsConfig:
    """Gestion de la configuration MLOps"""
    
    def __init__(self):
        self.config_dir = 'mlops_config'
        os.makedirs(self.config_dir, exist_ok=True)
        
        self.config = {
            'project': {
                'name': 'phishing-detection',
                'version': '1.0.0',
                'description': 'Deep Learning model for SMS and Email phishing detection',
                'authors': ['Data Science Team'],
                'created': datetime.now().isoformat()
            },
            'data': {
                'sources': {
                    'sms': os.path.join('data', 'sms_spam.csv'),
                    'email': os.path.join('data', 'email_dataset.csv')
                },
                'preprocessing': {
                    'lowercase': True,
                    'remove_urls': True,
                    'remove_emails': True,
                    'remove_special_chars': True,
                    'remove_stopwords': False,
                    'max_length': 128
                },
                'split': {
                    'train': 0.7,
                    'validation': 0.1,
                    'test': 0.2,
                    'random_seed': 42,
                    'stratify': True
                }
            },
            'model': {
                'architecture': 'RoBERTa-base',
                'pretrained_from': 'roberta-base',
                'num_labels': 2,
                'max_length': 128,
                'parameters': 125_000_000
            },
            'training': {
                'optimizer': 'AdamW',
                'learning_rate': 2e-5,
                'batch_size': 16,
                'epochs': 2,
                'warmup_steps': 100,
                'weight_decay': 0.01,
                'early_stopping_patience': 2,
                'device': 'cpu'
            },
            'evaluation': {
                'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
                'test_frequency': 'after_training',
                'cross_validation_folds': 5
            },
            'deployment': {
                'api_framework': 'FastAPI',
                'containerization': 'Docker',
                'port': 8000,
                'host': '0.0.0.0'
            },
            'mlops': {
                'reproducibility': {
                    'random_seed': 42,
                    'deterministic': True,
                    'version_control': 'git',
                    'data_versioning': False
                },
                'monitoring': {
                    'track_metrics': True,
                    'log_predictions': False,
                    'alert_on_drift': False
                },
                'ci_cd': {
                    'enabled': False,
                    'test_before_deploy': True,
                    'auto_rollback': False
                }
            }
        }
    
    def save_config(self, filename='mlops_config.json'):
        """Sauvegarde la configuration"""
        filepath = os.path.join(self.config_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"Configuration saved to {filepath}")
    
    def load_config(self, filename='mlops_config.json'):
        """Charge la configuration"""
        filepath = os.path.join(self.config_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.config = json.load(f)
            print(f"Configuration loaded from {filepath}")
        else:
            print(f"Config file not found: {filepath}")
    
    def get_config(self):
        """Retourne la configuration actuelle"""
        return self.config


class ExperimentTracker:
    """Suivi des expériences et versions du modèle"""
    
    def __init__(self):
        self.experiments_dir = 'experiments'
        os.makedirs(self.experiments_dir, exist_ok=True)
        self.experiments_file = os.path.join(self.experiments_dir, 'experiments.json')
        self.experiments = self._load_experiments()
    
    def _load_experiments(self):
        """Charge les expériences existantes"""
        if os.path.exists(self.experiments_file):
            with open(self.experiments_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_experiments(self):
        """Sauvegarde les expériences"""
        with open(self.experiments_file, 'w') as f:
            json.dump(self.experiments, f, indent=4)
    
    def log_experiment(self, experiment_name, config, metrics):
        """Enregistre une nouvelle expérience"""
        experiment = {
            'id': len(self.experiments) + 1,
            'name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'metrics': metrics,
            'git_commit': self._get_git_commit()
        }
        
        self.experiments.append(experiment)
        self._save_experiments()
        
        print(f"Experiment '{experiment_name}' logged with ID {experiment['id']}")
        return experiment['id']
    
    def _get_git_commit(self):
        """Récupère le hash du commit Git actuel"""
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
            return commit
        except:
            return 'unknown'
    
    def get_best_experiment(self, metric='accuracy'):
        """Trouve la meilleure expérience selon une métrique"""
        if not self.experiments:
            return None
        
        best = max(self.experiments, 
                  key=lambda x: x['metrics'].get(metric, 0))
        return best
    
    def compare_experiments(self, exp_ids):
        """Compare plusieurs expériences"""
        experiments = [exp for exp in self.experiments if exp['id'] in exp_ids]
        
        if not experiments:
            print("No experiments found with those IDs")
            return None
        
        comparison = {
            'experiments': []
        }
        
        for exp in experiments:
            comparison['experiments'].append({
                'id': exp['id'],
                'name': exp['name'],
                'metrics': exp['metrics']
            })
        
        return comparison


class ModelVersioning:
    """Gestion des versions du modèle"""
    
    def __init__(self):
        self.versions_dir = 'model_versions'
        os.makedirs(self.versions_dir, exist_ok=True)
        self.registry_file = os.path.join(self.versions_dir, 'registry.json')
        self.registry = self._load_registry()
    
    def _load_registry(self):
        """Charge le registre des versions"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {'models': []}
    
    def _save_registry(self):
        """Sauvegarde le registre"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=4)
    
    def register_model(self, model_path, data_type, metrics, experiment_id=None):
        """Enregistre une nouvelle version du modèle"""
        
        # Calculer le checksum du modèle
        model_file = os.path.join(model_path, 'pytorch_model.bin')
        if os.path.exists(model_file):
            checksum = self._calculate_checksum(model_file)
        else:
            checksum = 'unknown'
        
        version = {
            'version': len(self.registry['models']) + 1,
            'data_type': data_type,
            'path': model_path,
            'checksum': checksum,
            'metrics': metrics,
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'active'
        }
        
        self.registry['models'].append(version)
        self._save_registry()
        
        print(f"Model v{version['version']} registered for {data_type}")
        return version['version']
    
    def _calculate_checksum(self, filepath):
        """Calcule le checksum MD5 d'un fichier"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_latest_version(self, data_type):
        """Récupère la dernière version pour un type de données"""
        models = [m for m in self.registry['models'] 
                 if m['data_type'] == data_type and m['status'] == 'active']
        
        if not models:
            return None
        
        return max(models, key=lambda x: x['version'])
    
    def promote_to_production(self, version_number):
        """Marque une version comme production"""
        for model in self.registry['models']:
            if model['version'] == version_number:
                model['status'] = 'production'
                model['promoted_at'] = datetime.now().isoformat()
                self._save_registry()
                print(f"Model v{version_number} promoted to production")
                return True
        
        print(f"Model v{version_number} not found")
        return False


class ReproducibilityManager:
    """Gestion de la reproductibilité des expériences"""
    
    @staticmethod
    def set_seeds(seed=42):
        """Configure tous les seeds pour la reproductibilité"""
        import random
        import numpy as np
        import torch
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    @staticmethod
    def save_environment():
        """Sauvegarde l'environnement actuel"""
        env_info = {
            'timestamp': datetime.now().isoformat(),
            'python_version': None,
            'packages': []
        }
        
        try:
            import sys
            env_info['python_version'] = sys.version
            
            # Liste des packages installés
            result = subprocess.check_output(['pip', 'freeze']).decode('utf-8')
            env_info['packages'] = result.split('\n')
        except:
            pass
        
        with open('environment_snapshot.json', 'w') as f:
            json.dump(env_info, f, indent=4)
        
        print("Environment snapshot saved to environment_snapshot.json")


def initialize_mlops():
    """Initialise la configuration MLOps complète"""
    print("Initializing MLOps configuration...")
    
    # Configuration
    config = MLOpsConfig()
    config.save_config()
    
    # Experiment tracking
    tracker = ExperimentTracker()
    
    # Model versioning
    versioning = ModelVersioning()
    
    # Reproductibilité
    ReproducibilityManager.set_seeds(42)
    ReproducibilityManager.save_environment()
    
    print("MLOps configuration initialized successfully")
    
    return config, tracker, versioning


if __name__ == '__main__':
    # Initialiser la configuration MLOps
    config, tracker, versioning = initialize_mlops()
    
    # Afficher la configuration
    print("\nCurrent Configuration:")
    print(json.dumps(config.get_config(), indent=2))
