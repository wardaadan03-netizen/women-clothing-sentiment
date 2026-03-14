import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

class ModelEvaluator:
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_feature_importance(model, feature_names, class_names, top_n=15, save_path=None):
        """Plot top features for each class"""
        fig, axes = plt.subplots(1, len(class_names), figsize=(18, 6))
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        for idx, class_name in enumerate(class_names):
            class_coefs = model.coef_[idx]
            top_indices = np.argsort(class_coefs)[-top_n:]
            top_features = [feature_names[i] for i in top_indices]
            top_scores = class_coefs[top_indices]
            
            axes[idx].barh(top_features, top_scores, color=colors[idx])
            axes[idx].set_title(f'Top Features for {class_name}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Coefficient Value')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_training_history(history, save_path=None):
        """Plot training history (for neural networks if used)"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Train')
        plt.plot(history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()