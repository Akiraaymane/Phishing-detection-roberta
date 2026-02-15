import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        text = re.sub(r'http\S+|www\S+|https\S+', '', str(text))
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess(self, df):
        df['text_clean'] = df['text'].apply(self.clean_text)
        df = df[df['text_clean'].str.len() > 0]
        df = df.drop_duplicates(subset=['text_clean'])
        return df

def load_sms_data(file_path):
    df = pd.read_csv(file_path, encoding='latin-1')
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
        df = df[['label', 'text']]
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

def load_email_data(file_path):
    df = pd.read_csv(file_path)
    if 'label' not in df.columns:
        for col in ['spam', 'is_spam', 'class']:
            if col in df.columns:
                df = df.rename(columns={col: 'label'})
                break
    if 'text' not in df.columns:
        for col in ['message', 'email', 'content']:
            if col in df.columns:
                df = df.rename(columns={col: 'text'})
                break
    df['label'] = df['label'].apply(lambda x: 1 if x in [1, '1', 'spam'] else 0)
    return df

def split_data(df, test_size=0.2, val_size=0.1):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=val_size, stratify=train_df['label'], random_state=42)
    return train_df, val_df, test_df
