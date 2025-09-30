import os
import pandas as pd
import numpy as np
from data_preprocessing import DataPreprocessor
from model import FraudDetectionCNN
import config

def train_model():
    """Train the fraud detection model"""
    print("Starting fraud detection model training...")
    
    # Create directories if they don't exist
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Create synthetic data (in real scenario, load from CSV)
    print("Creating synthetic data...")
    df = preprocessor.create_synthetic_data(n_samples=10000)
    df.to_csv(f'{config.RAW_DATA_DIR}/returns_data.csv', index=False)
    
    # Prepare data
    print("Preprocessing data...")
    X, y = preprocessor.prepare_data(df)
    
    # Split data
    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Build model
    print("Building model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = FraudDetectionCNN(input_shape)
    model.build_model()
    model.compile_model()
    
    print("Model Summary:")
    model.summary()
    
    # Train model
    print("Training model...")
    history = model.model.fit(
        X_train, y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=model.get_callbacks(),
        verbose=1
    )
    
    # Save model
    model.model.save(f'{config.MODELS_DIR}/fraud_detection_model.h5')
    
    # Save preprocessor
    import joblib
    joblib.dump(preprocessor, f'{config.MODELS_DIR}/preprocessor.pkl')
    
    # Save test data for evaluation
    np.save(f'{config.PROCESSED_DATA_DIR}/X_test.npy', X_test)
    np.save(f'{config.PROCESSED_DATA_DIR}/y_test.npy', y_test)
    
    print("Training completed!")
    print(f"Model saved to: {config.MODELS_DIR}/fraud_detection_model.h5")
    
    return history

if __name__ == "__main__":
    train_model()