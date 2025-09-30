import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import config

class FraudDetectionCNN:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self):
        """Build CNN model for fraud detection"""
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=self.input_shape))
        
        # Convolutional blocks
        for i, filters in enumerate(config.CONV_FILTERS):
            model.add(layers.Conv1D(
                filters=filters,
                kernel_size=config.KERNEL_SIZE,
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}'
            ))
            model.add(layers.BatchNormalization(name=f'batch_norm_{i+1}'))
            model.add(layers.MaxPooling1D(
                pool_size=config.POOL_SIZE,
                name=f'maxpool_{i+1}'
            ))
            model.add(layers.Dropout(
                config.DROPOUT_RATE,
                name=f'dropout_conv_{i+1}'
            ))
        
        # Flatten for dense layers
        model.add(layers.GlobalAveragePooling1D(name='global_avg_pool'))
        
        # Dense layers
        for i, units in enumerate(config.DENSE_UNITS):
            model.add(layers.Dense(
                units=units,
                activation='relu',
                name=f'dense_{i+1}'
            ))
            model.add(layers.BatchNormalization(name=f'batch_norm_dense_{i+1}'))
            model.add(layers.Dropout(
                config.DROPOUT_RATE,
                name=f'dropout_dense_{i+1}'
            ))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        
        self.model = model
        return model
    
    def compile_model(self):
        """Compile the model"""
        optimizer = optimizers.Adam(learning_rate=config.LEARNING_RATE)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
    
    def get_callbacks(self):
        """Get training callbacks"""
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            callbacks.ModelCheckpoint(
                filepath=f'{config.MODELS_DIR}/best_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        return callbacks_list
    
    def summary(self):
        """Print model summary"""
        if self.model:
            return self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")