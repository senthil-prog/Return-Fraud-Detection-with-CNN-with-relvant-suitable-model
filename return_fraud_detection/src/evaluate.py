import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
import joblib
import config

def evaluate_model():
    """Evaluate the trained model"""
    print("Loading model and test data...")
    
    # Load model
    model = tf.keras.models.load_model(f'{config.MODELS_DIR}/fraud_detection_model.h5')
    
    # Load test data
    X_test = np.load(f'{config.PROCESSED_DATA_DIR}/X_test.npy')
    y_test = np.load(f'{config.PROCESSED_DATA_DIR}/y_test.npy')
    
    print(f"Test set shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Make predictions
    print("Making predictions...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # ROC AUC score
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score: {auc_score:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{config.MODELS_DIR}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{config.MODELS_DIR}/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance analysis (approximate)
    print("\nGenerating feature importance analysis...")
    
    return {
        'auc_score': auc_score,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': cm
    }

if __name__ == "__main__":
    results = evaluate_model()