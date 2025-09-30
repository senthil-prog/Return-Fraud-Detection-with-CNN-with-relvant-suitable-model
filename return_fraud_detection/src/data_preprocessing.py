import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import config

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_synthetic_data(self, n_samples=10000):
        """Create synthetic return fraud data"""
        np.random.seed(42)
        
        # Generate synthetic features
        data = {
            'customer_id': range(1, n_samples + 1),
            'return_frequency': np.random.exponential(2, n_samples),
            'return_amount': np.random.lognormal(4, 1, n_samples),
            'days_since_purchase': np.random.exponential(10, n_samples),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_samples),
            'customer_age': np.random.normal(35, 12, n_samples),
            'order_value': np.random.lognormal(5, 0.5, n_samples),
            'return_reason': np.random.choice(['Defective', 'Size', 'Changed Mind', 'Wrong Item'], n_samples),
            'refund_method': np.random.choice(['Credit Card', 'Store Credit', 'Cash'], n_samples),
            'customer_history': np.random.exponential(5, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create fraud labels based on suspicious patterns
        fraud_conditions = (
            (df['return_frequency'] > 5) |
            (df['return_amount'] > 1000) |
            (df['days_since_purchase'] < 1) |
            ((df['return_frequency'] > 3) & (df['customer_history'] < 2))
        )
        
        df['is_fraud'] = fraud_conditions.astype(int)
        
        # Add some noise to make it more realistic
        fraud_noise = np.random.random(n_samples) < 0.1
        df.loc[fraud_noise, 'is_fraud'] = 1 - df.loc[fraud_noise, 'is_fraud']
        
        return df
    
    def preprocess_features(self, df):
        """Preprocess features for CNN input"""
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['product_category', 'return_reason', 'refund_method']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        # Select features for model
        feature_columns = [col for col in config.FEATURES if col in df_processed.columns]
        X = df_processed[feature_columns].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled
    
    def create_sequences(self, data, sequence_length=config.SEQUENCE_LENGTH):
        """Create sequences for CNN input"""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:(i + sequence_length)])
        return np.array(sequences)
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Preprocess features
        X = self.preprocess_features(df)
        y = df['is_fraud'].values
        
        # Create sequences
        X_sequences = self.create_sequences(X)
        y_sequences = y[config.SEQUENCE_LENGTH-1:]
        
        return X_sequences, y_sequences
    
    def split_data(self, X, y):
        """Split data into train, validation, and test sets"""
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=config.TEST_SPLIT, random_state=42, stratify=y
        )
        
        # Second split: separate train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=config.VALIDATION_SPLIT/(1-config.TEST_SPLIT), 
            random_state=42, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test