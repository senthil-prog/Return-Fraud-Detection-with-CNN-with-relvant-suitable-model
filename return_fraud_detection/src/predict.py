import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import config

class FraudPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.load_model()
    
    def load_model(self):
        """Load trained model and preprocessor"""
        try:
            self.model = tf.keras.models.load_model(f'{config.MODELS_DIR}/fraud_detection_model.h5')
            self.preprocessor = joblib.load(f'{config.MODELS_DIR}/preprocessor.pkl')
            print("Model and preprocessor loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def predict_single(self, return_data):
        """Predict fraud for a single return"""
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model not loaded properly")
        
        # Convert to DataFrame
        if isinstance(return_data, dict):
            df = pd.DataFrame([return_data])
        else:
            df = return_data
        
        # Preprocess
        X = self.preprocessor.preprocess_features(df)
        
        # Create sequence (for single prediction, repeat the data)
        X_seq = np.repeat(X.reshape(1, 1, -1), config.SEQUENCE_LENGTH, axis=1)
        
        # Predict
        probability = self.model.predict(X_seq)[0][0]
        prediction = int(probability > 0.5)
        
        return {
            'is_fraud': prediction,
            'fraud_probability': float(probability),
            'risk_level': self._get_risk_level(probability)
        }
    
    def predict_batch(self, returns_data):
        """Predict fraud for multiple returns"""
        if isinstance(returns_data, dict):
            df = pd.DataFrame(returns_data)
        else:
            df = returns_data
        
        results = []
        for idx, row in df.iterrows():
            result = self.predict_single(row.to_dict())
            result['return_id'] = idx
            results.append(result)
        
        return results
    
    def _get_risk_level(self, probability):
        """Determine risk level based on probability"""
        if probability < 0.3:
            return 'Low'
        elif probability < 0.7:
            return 'Medium'
        else:
            return 'High'

def example_prediction():
    """Example of how to use the predictor"""
    predictor = FraudPredictor()
    
    # Example return data
    sample_return = {
        'return_frequency': 3.5,
        'return_amount': 250.0,
        'days_since_purchase': 2.0,
        'product_category': 'Electronics',
        'customer_age': 28.0,
        'order_value': 300.0,
        'return_reason': 'Defective',
        'refund_method': 'Credit Card',
        'customer_history': 1.5
    }
    
    result = predictor.predict_single(sample_return)
    
    print("Fraud Detection Result:")
    print(f"Is Fraud: {result['is_fraud']}")
    print(f"Fraud Probability: {result['fraud_probability']:.4f}")
    print(f"Risk Level: {result['risk_level']}")
    
    return result

if __name__ == "__main__":
    example_prediction()