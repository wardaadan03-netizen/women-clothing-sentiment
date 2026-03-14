import pandas as pd
import numpy as np
import json
import os

def load_data(filepath):
    """Load and prepare the dataset"""
    df = pd.read_csv(filepath)
    
    # Drop unnamed column if exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    return df

def create_sentiment_labels(df, rating_col='Rating'):
    """Add sentiment column based on ratings"""
    def sentiment_label(rating):
        if rating >= 4:
            return 'Positive'
        elif rating == 3:
            return 'Neutral'
        else:
            return 'Negative'
    
    df['sentiment'] = df[rating_col].apply(sentiment_label)
    return df

def get_basic_stats(df):
    """Get basic statistics about the dataset"""
    stats = {
        'total_reviews': len(df),
        'avg_rating': df['Rating'].mean(),
        'rating_distribution': df['Rating'].value_counts().sort_index().to_dict(),
        'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
        'missing_values': df.isnull().sum().to_dict()
    }
    return stats

def save_json(data, filepath):
    """Save data as JSON"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filepath):
    """Load JSON data"""
    with open(filepath, 'r') as f:
        return json.load(f)

def ensure_dir(directory):
    """Ensure directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)