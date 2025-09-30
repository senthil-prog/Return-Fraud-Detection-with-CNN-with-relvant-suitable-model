import os

# Data paths
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(DATA_DIR, "models")

# Model parameters
SEQUENCE_LENGTH = 30
FEATURES = [
    'return_frequency', 'return_amount', 'days_since_purchase',
    'product_category', 'customer_age', 'order_value',
    'return_reason', 'refund_method', 'customer_history'
]

# Training parameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2

# Model architecture
CONV_FILTERS = [32, 64, 128]
KERNEL_SIZE = 3
POOL_SIZE = 2
DROPOUT_RATE = 0.3
DENSE_UNITS = [256, 128, 64]