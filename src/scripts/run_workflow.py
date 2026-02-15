"""
Script d'automatisation du workflow complet du projet
Exécute toutes les étapes: EDA, training, testing, comparison
"""

import os
import subprocess
import argparse
import time
from datetime import datetime


class WorkflowAutomation:
    """Automatise le workflow complet du projet"""
    
    def __init__(self, data_type='sms'):
        self.data_type = data_type
        self.data_path = os.path.join('data', f'{data_type}_spam.csv') if data_type == 'sms' else os.path.join('data', 'email_dataset.csv')
        self.model_dir = os.path.join('models', f'roberta-{data_type}')
        self.start_time = None
        self.log_file = f'workflow_log_{data_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    
    def log(self, message):
        """Enregistre un message dans le log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def run_command(self, command, description):
        """Exécute une commande et log le résultat"""
        self.log(f"\nStarting: {description}")
        self.log(f"Command: {command}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(command, shell=True, check=True, 
                                   capture_output=True, text=True)
            elapsed = time.time() - start_time
            self.log(f"Completed in {elapsed:.2f} seconds")
            return True
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            self.log(f"FAILED after {elapsed:.2f} seconds")
            self.log(f"Error: {e.stderr}")
            return False
    
    def step1_eda(self):
        """Étape 1: Analyse exploratoire des données"""
        self.log("\n" + "="*60)
        self.log("STEP 1: EXPLORATORY DATA ANALYSIS")
        self.log("="*60)
        
        command = f"python src/scripts/eda_analysis.py --data_type {self.data_type} --data_path {self.data_path}"
        return self.run_command(command, "EDA Analysis")
    
    def step2_train_local(self):
        """Étape 2: Entraînement en local"""
        self.log("\n" + "="*60)
        self.log("STEP 2: LOCAL TRAINING")
        self.log("="*60)
        
        command = f"python src/training/train.py --data_type {self.data_type} --data_path {self.data_path} --output_dir {self.model_dir} --epochs 2 --batch_size 16"
        return self.run_command(command, "Local Training")
    
    def step3_test_model(self):
        """Étape 3: Test du modèle"""
        self.log("\n" + "="*60)
        self.log("STEP 3: MODEL TESTING")
        self.log("="*60)
        
        if not os.path.exists(self.model_dir):
            self.log(f"Model directory {self.model_dir} not found. Skipping testing.")
            return False
        
        command = f"python src/training/test.py --data_type {self.data_type} --data_path {self.data_path} --model_path {self.model_dir}"
        return self.run_command(command, "Model Testing")
    
    def step4_cross_validation(self, n_splits=3):
        """Étape 4: Validation croisée"""
        self.log("\n" + "="*60)
        self.log(f"STEP 4: CROSS-VALIDATION ({n_splits} folds)")
        self.log("="*60)
        
        command = f"python src/scripts/cross_validation.py --data_type {self.data_type} --data_path {self.data_path} --n_splits {n_splits} --epochs 1 --batch_size 16"
        return self.run_command(command, f"Cross-Validation ({n_splits} folds)")
    
    def step5_train_docker(self):
        """Étape 5: Entraînement avec Docker"""
        self.log("\n" + "="*60)
        self.log("STEP 5: DOCKER TRAINING")
        self.log("="*60)
        
        service_name = f"train-{self.data_type}"
        command = f"docker-compose up {service_name}"
        return self.run_command(command, "Docker Training")
    
    def step6_compare_environments(self):
        """Étape 6: Comparaison des environnements"""
        self.log("\n" + "="*60)
        self.log("STEP 6: ENVIRONMENT COMPARISON")
        self.log("="*60)
        
        local_metrics = os.path.join('results', f"local_{self.data_type}_metrics.json")
        docker_metrics = os.path.join('results', f"docker_{self.data_type}_metrics.json")
        
        if not os.path.exists(local_metrics):
            self.log(f"Local metrics not found: {local_metrics}")
            return False
        
        if not os.path.exists(docker_metrics):
            self.log(f"Docker metrics not found: {docker_metrics}")
            return False
        
        
        command = f"python src/scripts/compare_environments.py --data_type {self.data_type}"
        return self.run_command(command, "Environment Comparison")
    
    def step7_test_api(self):
        """Étape 7: Test de l'API"""
        self.log("\n" + "="*60)
        self.log("STEP 7: API TESTING")
        self.log("="*60)
        
        command = "pytest src/api/test_api.py -v"
        return self.run_command(command, "API Unit Tests")
    
    def run_full_workflow(self, skip_docker=False, skip_cv=False):
        """Exécute le workflow complet"""
        self.start_time = time.time()
        
        self.log("="*60)
        self.log(f"STARTING FULL WORKFLOW FOR {self.data_type.upper()}")
        self.log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("="*60)
        
        steps = [
            ('EDA', self.step1_eda),
            ('Local Training', self.step2_train_local),
            ('Model Testing', self.step3_test_model),
        ]
        
        if not skip_cv:
            steps.append(('Cross-Validation', lambda: self.step4_cross_validation(n_splits=3)))
        
        if not skip_docker:
            steps.extend([
                ('Docker Training', self.step5_train_docker),
                ('Environment Comparison', self.step6_compare_environments),
            ])
        
        steps.append(('API Testing', self.step7_test_api))
        
        results = {}
        
        for step_name, step_func in steps:
            success = step_func()
            results[step_name] = 'SUCCESS' if success else 'FAILED'
            
            if not success:
                self.log(f"\nStep '{step_name}' failed. Continuing with next steps...")
        
        total_time = time.time() - self.start_time
        
        self.log("\n" + "="*60)
        self.log("WORKFLOW SUMMARY")
        self.log("="*60)
        
        for step_name, status in results.items():
            self.log(f"{step_name:25s} : {status}")
        
        self.log(f"\nTotal execution time: {total_time/60:.2f} minutes")
        self.log(f"Log saved to: {self.log_file}")
        self.log("="*60)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Automate the complete ML workflow')
    parser.add_argument('--data_type', type=str, default='sms', choices=['sms', 'email'],
                       help='Type of data to process (default: sms)')
    parser.add_argument('--skip_docker', action='store_true',
                       help='Skip Docker training and comparison')
    parser.add_argument('--skip_cv', action='store_true',
                       help='Skip cross-validation (saves time)')
    parser.add_argument('--step', type=str, choices=['eda', 'train', 'test', 'cv', 'docker', 'compare', 'api_test', 'all'],
                       default='all', help='Run specific step or all')
    args = parser.parse_args()
    
    workflow = WorkflowAutomation(data_type=args.data_type)
    
    if args.step == 'all':
        workflow.run_full_workflow(skip_docker=args.skip_docker, skip_cv=args.skip_cv)
    elif args.step == 'eda':
        workflow.step1_eda()
    elif args.step == 'train':
        workflow.step2_train_local()
    elif args.step == 'test':
        workflow.step3_test_model()
    elif args.step == 'cv':
        workflow.step4_cross_validation()
    elif args.step == 'docker':
        workflow.step5_train_docker()
    elif args.step == 'compare':
        workflow.step6_compare_environments()
    elif args.step == 'api_test':
        workflow.step7_test_api()


if __name__ == '__main__':
    main()
