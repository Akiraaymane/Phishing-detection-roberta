"""
Script de comparaison des performances entre environnement local et Docker
Analyse les métriques d'entraînement et génère un rapport comparatif
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


class EnvironmentComparator:
    """Compare les résultats d'entraînement entre environnements local et Docker"""
    
    def __init__(self):
        self.output_dir = os.path.join('results', 'comparison_results')
        os.makedirs(self.output_dir, exist_ok=True)
        self.comparison_data = []
    
    def load_metrics(self, env_type, data_type):
        """Charge les métriques depuis les fichiers JSON"""
        filename = os.path.join('results', f"{env_type}_{data_type}_metrics.json")
        
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found")
            return None
        
        with open(filename, 'r') as f:
            return json.load(f)
    
    def compare(self, data_type='sms'):
        """Compare les performances local vs Docker pour un type de données"""
        print(f"\nComparing {data_type.upper()} results between LOCAL and DOCKER environments\n")
        
        local_metrics = self.load_metrics('local', data_type)
        docker_metrics = self.load_metrics('docker', data_type)
        
        if not local_metrics:
            print(f"Local metrics for {data_type} not found. Please train locally first.")
            return
        
        if not docker_metrics:
            print(f"Docker metrics for {data_type} not found. Please train with Docker first.")
            return
        
        # Créer le tableau de comparaison
        comparison = {
            'Metric': ['Training Time (min)', 'Accuracy (%)', 'F1-Score', 
                      'Precision', 'Recall', 'Epochs', 'Batch Size'],
            'Local': [
                local_metrics['training_time_minutes'],
                local_metrics['accuracy'] * 100,
                local_metrics['f1_score'],
                local_metrics['precision'],
                local_metrics['recall'],
                local_metrics['epochs'],
                local_metrics['batch_size']
            ],
            'Docker': [
                docker_metrics['training_time_minutes'],
                docker_metrics['accuracy'] * 100,
                docker_metrics['f1_score'],
                docker_metrics['precision'],
                docker_metrics['recall'],
                docker_metrics['epochs'],
                docker_metrics['batch_size']
            ]
        }
        
        df_comparison = pd.DataFrame(comparison)
        df_comparison['Difference'] = df_comparison['Docker'] - df_comparison['Local']
        df_comparison['Difference (%)'] = (df_comparison['Difference'] / df_comparison['Local']) * 100
        
        print("\nComparison Table:")
        print(df_comparison.to_string(index=False))
        
        # Sauvegarder le tableau
        df_comparison.to_csv(f'{self.output_dir}/comparison_{data_type}.csv', index=False)
        
        # Générer les graphiques
        self._generate_comparison_plots(local_metrics, docker_metrics, data_type)
        
        # Générer le rapport
        self._generate_report(local_metrics, docker_metrics, df_comparison, data_type)
        
        return df_comparison
    
    def _generate_comparison_plots(self, local_metrics, docker_metrics, data_type):
        """Génère les graphiques de comparaison"""
        
        # Graphique 1: Comparaison des métriques de performance
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        metric_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            row = idx // 2
            col = idx % 2
            
            local_val = local_metrics[metric]
            docker_val = docker_metrics[metric]
            
            bars = axes[row, col].bar(['Local', 'Docker'], 
                                     [local_val, docker_val],
                                     color=['#3498db', '#e74c3c'])
            axes[row, col].set_ylabel('Score')
            axes[row, col].set_title(f'{name} Comparison')
            axes[row, col].set_ylim([0, 1])
            axes[row, col].grid(axis='y', alpha=0.3)
            
            # Ajouter les valeurs sur les barres
            for bar in bars:
                height = bar.get_height()
                axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.4f}',
                                   ha='center', va='bottom')
        
        plt.suptitle(f'Performance Comparison - {data_type.upper()}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/performance_comparison_{data_type}.png', dpi=300)
        plt.close()
        
        # Graphique 2: Comparaison du temps d'entraînement
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        times = [local_metrics['training_time_minutes'], 
                docker_metrics['training_time_minutes']]
        bars = ax.bar(['Local', 'Docker'], times, color=['#3498db', '#e74c3c'])
        ax.set_ylabel('Time (minutes)')
        ax.set_title(f'Training Time Comparison - {data_type.upper()}')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f} min',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/time_comparison_{data_type}.png', dpi=300)
        plt.close()
        
        print(f"\nPlots saved to {self.output_dir}/")
    
    def _generate_report(self, local_metrics, docker_metrics, df_comparison, data_type):
        """Génère un rapport texte de comparaison"""
        
        report_file = f'{self.output_dir}/comparison_report_{data_type}.txt'
        
        with open(report_file, 'w') as f:
            f.write("ENVIRONMENT COMPARISON REPORT\n")
            f.write(f"Data Type: {data_type.upper()}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("LOCAL ENVIRONMENT:\n")
            f.write(f"  Training Time: {local_metrics['training_time_minutes']:.2f} minutes\n")
            f.write(f"  Accuracy: {local_metrics['accuracy']*100:.2f}%\n")
            f.write(f"  F1-Score: {local_metrics['f1_score']:.4f}\n")
            f.write(f"  Precision: {local_metrics['precision']:.4f}\n")
            f.write(f"  Recall: {local_metrics['recall']:.4f}\n\n")
            
            f.write("DOCKER ENVIRONMENT:\n")
            f.write(f"  Training Time: {docker_metrics['training_time_minutes']:.2f} minutes\n")
            f.write(f"  Accuracy: {docker_metrics['accuracy']*100:.2f}%\n")
            f.write(f"  F1-Score: {docker_metrics['f1_score']:.4f}\n")
            f.write(f"  Precision: {docker_metrics['precision']:.4f}\n")
            f.write(f"  Recall: {docker_metrics['recall']:.4f}\n\n")
            
            f.write("DIFFERENCES:\n")
            time_diff = docker_metrics['training_time_minutes'] - local_metrics['training_time_minutes']
            acc_diff = (docker_metrics['accuracy'] - local_metrics['accuracy']) * 100
            
            f.write(f"  Training Time: {time_diff:+.2f} minutes ")
            f.write(f"({(time_diff/local_metrics['training_time_minutes'])*100:+.1f}%)\n")
            f.write(f"  Accuracy: {acc_diff:+.2f}%\n\n")
            
            f.write("ANALYSIS:\n")
            if abs(acc_diff) < 1.0:
                f.write("  Performance: Similar accuracy between environments (expected)\n")
            elif acc_diff > 0:
                f.write("  Performance: Docker environment shows slightly better accuracy\n")
            else:
                f.write("  Performance: Local environment shows slightly better accuracy\n")
            
            if time_diff < 0:
                f.write("  Speed: Docker is faster (unexpected, may indicate resource optimization)\n")
            elif time_diff > 5:
                f.write("  Speed: Docker is slower (expected due to containerization overhead)\n")
            else:
                f.write("  Speed: Similar training time between environments\n")
            
            f.write("\nREPRODUCIBILITY:\n")
            f.write("  Both environments use the same:\n")
            f.write("    - Model architecture (RoBERTa Transformer)\n")
            f.write("    - Hyperparameters (learning rate, batch size, epochs)\n")
            f.write("    - Random seeds for data splitting\n")
            f.write("    - Preprocessing pipeline\n")
            f.write("  This ensures reproducible results across environments.\n")
        
        print(f"Report saved to {report_file}")
    
    def compare_all(self):
        """Compare tous les types de données disponibles"""
        data_types = []
        
        # Détecter les types de données disponibles
        for data_type in ['sms', 'email']:
            local_file = os.path.join('results', f"local_{data_type}_metrics.json")
            docker_file = os.path.join('results', f"docker_{data_type}_metrics.json")
            if os.path.exists(local_file) or os.path.exists(docker_file):
                data_types.append(data_type)
        
        if not data_types:
            print("No metrics files found. Please train models first.")
            return
        
        print(f"Found data types: {', '.join(data_types)}")
        
        for data_type in data_types:
            self.compare(data_type)
        
        print(f"\nAll comparison results saved to {self.output_dir}/")


def main():
    """Point d'entrée principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare local vs Docker training results')
    parser.add_argument('--data_type', type=str, choices=['sms', 'email', 'all'],
                       default='all', help='Type of data to compare (default: all)')
    args = parser.parse_args()
    
    comparator = EnvironmentComparator()
    
    if args.data_type == 'all':
        comparator.compare_all()
    else:
        comparator.compare(args.data_type)


if __name__ == '__main__':
    main()
