"""
Script d'Analyse Exploratoire des Données (EDA)
Génère des statistiques et visualisations complètes sur les datasets
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from src.preprocessing import TextPreprocessor, load_sms_data, load_email_data


class EDAAnalyzer:
    """Classe pour l'analyse exploratoire des données"""
    
    def __init__(self, data_type):
        self.data_type = data_type
        self.output_dir = f'eda_results_{data_type}'
        os.makedirs(self.output_dir, exist_ok=True)
        self.preprocessor = TextPreprocessor()
    
    def analyze(self, df):
        """Analyse complète du dataset"""
        print(f"\n{'='*60}")
        print(f"EXPLORATORY DATA ANALYSIS - {self.data_type.upper()}")
        print(f"{'='*60}\n")
        
        # 1. Statistiques générales
        self._basic_statistics(df)
        
        # 2. Distribution des classes
        self._class_distribution(df)
        
        # 3. Analyse de la longueur des textes
        self._text_length_analysis(df)
        
        # 4. Analyse des mots
        self._word_analysis(df)
        
        # 5. Word clouds
        self._generate_wordclouds(df)
        
        # 6. N-grams
        self._ngram_analysis(df)
        
        # 7. Caractères spéciaux
        self._special_chars_analysis(df)
        
        # 8. Corrélations
        self._feature_correlations(df)
        
        print(f"\n{'='*60}")
        print(f"EDA COMPLETE! Results saved to {self.output_dir}/")
        print(f"{'='*60}\n")
    
    def _basic_statistics(self, df):
        """Statistiques de base"""
        print("=== BASIC STATISTICS ===\n")
        print(f"Total samples: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nMissing values:")
        print(df.isnull().sum())
        print(f"\nDuplicate rows: {df.duplicated().sum()}")
        
        # Sauvegarder
        stats = {
            'total_samples': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': int(df.duplicated().sum()),
            'label_distribution': df['label'].value_counts().to_dict()
        }
        
        with open(f'{self.output_dir}/basic_stats.txt', 'w') as f:
            f.write(f"Dataset: {self.data_type.upper()}\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Legitimate: {(df['label']==0).sum()}\n")
            f.write(f"Phishing: {(df['label']==1).sum()}\n")
            f.write(f"Missing values: {df.isnull().sum().sum()}\n")
            f.write(f"Duplicates: {df.duplicated().sum()}\n")
    
    def _class_distribution(self, df):
        """Distribution des classes"""
        print("\n=== CLASS DISTRIBUTION ===\n")
        
        class_counts = df['label'].value_counts()
        print(f"Legitimate (0): {class_counts[0]} ({class_counts[0]/len(df)*100:.2f}%)")
        print(f"Phishing (1): {class_counts[1]} ({class_counts[1]/len(df)*100:.2f}%)")
        
        # Graphique
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart
        axes[0].bar(['Legitimate', 'Phishing'], 
                    [class_counts[0], class_counts[1]], 
                    color=['#2ecc71', '#e74c3c'])
        axes[0].set_ylabel('Count')
        axes[0].set_title('Class Distribution')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add counts on bars
        for i, v in enumerate([class_counts[0], class_counts[1]]):
            axes[0].text(i, v, str(v), ha='center', va='bottom')
        
        # Pie chart
        axes[1].pie([class_counts[0], class_counts[1]], 
                    labels=['Legitimate', 'Phishing'],
                    autopct='%1.1f%%', 
                    colors=['#2ecc71', '#e74c3c'],
                    startangle=90)
        axes[1].set_title('Class Proportion')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/class_distribution.png', dpi=300)
        plt.close()
    
    def _text_length_analysis(self, df):
        """Analyse de la longueur des textes"""
        print("\n=== TEXT LENGTH ANALYSIS ===\n")
        
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        
        # Statistiques par classe
        for label, name in [(0, 'Legitimate'), (1, 'Phishing')]:
            subset = df[df['label'] == label]
            print(f"\n{name}:")
            print(f"  Avg length: {subset['text_length'].mean():.2f} chars")
            print(f"  Avg words: {subset['word_count'].mean():.2f} words")
            print(f"  Min length: {subset['text_length'].min()}")
            print(f"  Max length: {subset['text_length'].max()}")
        
        # Graphiques
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Distribution de la longueur par classe
        for label, name, color in [(0, 'Legitimate', '#2ecc71'), (1, 'Phishing', '#e74c3c')]:
            subset = df[df['label'] == label]
            axes[0, 0].hist(subset['text_length'], bins=50, alpha=0.6, label=name, color=color)
        axes[0, 0].set_xlabel('Text Length (characters)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Text Length Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Distribution du nombre de mots
        for label, name, color in [(0, 'Legitimate', '#2ecc71'), (1, 'Phishing', '#e74c3c')]:
            subset = df[df['label'] == label]
            axes[0, 1].hist(subset['word_count'], bins=50, alpha=0.6, label=name, color=color)
        axes[0, 1].set_xlabel('Word Count')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Word Count Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Boxplot longueur
        df_plot = df.copy()
        df_plot['label_name'] = df_plot['label'].map({0: 'Legitimate', 1: 'Phishing'})
        axes[1, 0].boxplot([df[df['label']==0]['text_length'], 
                            df[df['label']==1]['text_length']],
                           labels=['Legitimate', 'Phishing'])
        axes[1, 0].set_ylabel('Text Length (characters)')
        axes[1, 0].set_title('Text Length by Class')
        axes[1, 0].grid(alpha=0.3)
        
        # Boxplot mots
        axes[1, 1].boxplot([df[df['label']==0]['word_count'], 
                            df[df['label']==1]['word_count']],
                           labels=['Legitimate', 'Phishing'])
        axes[1, 1].set_ylabel('Word Count')
        axes[1, 1].set_title('Word Count by Class')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/text_length_analysis.png', dpi=300)
        plt.close()
    
    def _word_analysis(self, df):
        """Analyse des mots les plus fréquents"""
        print("\n=== WORD ANALYSIS ===\n")
        
        # Mots les plus fréquents par classe
        for label, name in [(0, 'Legitimate'), (1, 'Phishing')]:
            texts = ' '.join(df[df['label']==label]['text'].values)
            words = re.findall(r'\b[a-zA-Z]+\b', texts.lower())
            word_freq = Counter(words).most_common(20)
            
            print(f"\nTop 20 words in {name}:")
            for word, count in word_freq[:10]:
                print(f"  {word}: {count}")
        
        # Graphique
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, (label, name, color) in enumerate([
            (0, 'Legitimate', '#2ecc71'), 
            (1, 'Phishing', '#e74c3c')
        ]):
            texts = ' '.join(df[df['label']==label]['text'].values)
            words = re.findall(r'\b[a-zA-Z]+\b', texts.lower())
            word_freq = Counter(words).most_common(15)
            
            words_list = [w[0] for w in word_freq]
            counts = [w[1] for w in word_freq]
            
            axes[idx].barh(words_list, counts, color=color)
            axes[idx].set_xlabel('Frequency')
            axes[idx].set_title(f'Top 15 Words - {name}')
            axes[idx].invert_yaxis()
            axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/word_frequency.png', dpi=300)
        plt.close()
    
    def _generate_wordclouds(self, df):
        """Génère des word clouds"""
        print("\n=== GENERATING WORD CLOUDS ===")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, (label, name, colormap) in enumerate([
            (0, 'Legitimate', 'Greens'), 
            (1, 'Phishing', 'Reds')
        ]):
            texts = ' '.join(df[df['label']==label]['text'].values)
            
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap=colormap,
                max_words=100
            ).generate(texts)
            
            axes[idx].imshow(wordcloud, interpolation='bilinear')
            axes[idx].set_title(f'Word Cloud - {name}', fontsize=16)
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/wordclouds.png', dpi=300)
        plt.close()
        print("Word clouds generated!")
    
    def _ngram_analysis(self, df):
        """Analyse des n-grams"""
        print("\n=== N-GRAM ANALYSIS ===")
        
        def get_ngrams(text, n=2):
            words = text.lower().split()
            return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for row, n in enumerate([2, 3]):  # Bigrams et trigrams
            for col, (label, name, color) in enumerate([
                (0, 'Legitimate', '#2ecc71'), 
                (1, 'Phishing', '#e74c3c')
            ]):
                texts = ' '.join(df[df['label']==label]['text'].values)
                ngrams = get_ngrams(texts, n)
                ngram_freq = Counter(ngrams).most_common(10)
                
                ngram_list = [ng[0] for ng in ngram_freq]
                counts = [ng[1] for ng in ngram_freq]
                
                axes[row, col].barh(ngram_list, counts, color=color)
                axes[row, col].set_xlabel('Frequency')
                axes[row, col].set_title(f'Top 10 {n}-grams - {name}')
                axes[row, col].invert_yaxis()
                axes[row, col].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/ngrams_analysis.png', dpi=300)
        plt.close()
    
    def _special_chars_analysis(self, df):
        """Analyse des caractères spéciaux"""
        print("\n=== SPECIAL CHARACTERS ANALYSIS ===")
        
        df['num_exclamation'] = df['text'].str.count('!')
        df['num_question'] = df['text'].str.count('\?')
        df['num_uppercase'] = df['text'].str.count('[A-Z]')
        df['num_digits'] = df['text'].str.count('\d')
        df['num_urls'] = df['text'].str.count(r'http[s]?://|www\.')
        
        features = ['num_exclamation', 'num_question', 'num_uppercase', 'num_digits', 'num_urls']
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(features):
            for label, name, color in [(0, 'Legitimate', '#2ecc71'), (1, 'Phishing', '#e74c3c')]:
                subset = df[df['label'] == label][feature]
                axes[idx].hist(subset, bins=20, alpha=0.6, label=name, color=color)
            
            axes[idx].set_xlabel(feature.replace('_', ' ').title())
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'Distribution of {feature.replace("_", " ").title()}')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/special_chars.png', dpi=300)
        plt.close()
    
    def _feature_correlations(self, df):
        """Analyse des corrélations entre features"""
        print("\n=== FEATURE CORRELATIONS ===")
        
        # Créer des features numériques
        numeric_df = df[['label', 'text_length', 'word_count', 'num_exclamation', 
                         'num_question', 'num_uppercase', 'num_digits', 'num_urls']].copy()
        
        # Matrice de corrélation
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/correlation_matrix.png', dpi=300)
        plt.close()
        
        print(f"\nCorrelations with label:")
        print(corr_matrix['label'].sort_values(ascending=False))


def main():
    parser = argparse.ArgumentParser(description='Exploratory Data Analysis')
    parser.add_argument('--data_type', type=str, required=True, choices=['sms', 'email'],
                       help='Type of data: sms or email')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to CSV dataset')
    args = parser.parse_args()
    
    # Charger les données
    print(f"Loading {args.data_type} data...")
    if args.data_type == 'sms':
        df = load_sms_data(args.data_path)
    else:
        df = load_email_data(args.data_path)
    
    # Créer l'analyseur
    analyzer = EDAAnalyzer(args.data_type)
    
    # Lancer l'analyse
    analyzer.analyze(df)


if __name__ == '__main__':
    main()
