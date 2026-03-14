
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

class SentimentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=5,
            max_df=0.8,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.is_trained = False
    
    def prepare_features(self, texts, fit_vectorizer=False):
        """Convert texts to TF-IDF features"""
        if fit_vectorizer:
            return self.vectorizer.fit_transform(texts)
        return self.vectorizer.transform(texts)
    
    def train(self, X_text, y, test_size=0.2, random_state=42):
        """Train the sentiment analysis model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y, test_size=test_size, 
            random_state=random_state, stratify=y
        )
        
        # Vectorize
        X_train_tfidf = self.prepare_features(X_train, fit_vectorizer=True)
        X_test_tfidf = self.prepare_features(X_test)
        
        # Train model
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_tfidf, y_train, cv=5)
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def predict(self, text):
        """Predict sentiment for a single text"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Vectorize
        features = self.prepare_features([text])
        
        # Predict
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        return {
            'sentiment': prediction,
            'confidence': {
                class_name: prob 
                for class_name, prob in zip(self.model.classes_, probabilities)
            }
        }
    
    def predict_batch(self, texts):
        """Predict sentiment for multiple texts"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        features = self.prepare_features(texts)
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        results = []
        for i, text in enumerate(texts):
            results.append({
                'text': text,
                'sentiment': predictions[i],
                'confidence': {
                    class_name: probabilities[i][j]
                    for j, class_name in enumerate(self.model.classes_)
                }
            })
        
        return results
    
    def save_model(self, model_path, vectorizer_path):
        """Save trained model and vectorizer"""
        if not self.is_trained:
            raise ValueError("No trained model to save!")
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load_model(self, model_path, vectorizer_path):
        """Load trained model and vectorizer"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        self.is_trained = True