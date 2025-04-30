import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

class BackdoorAttacks:
    def __init__(self, poison_ratio=0.1):
        self.poison_ratio = poison_ratio
        # Load and preprocess CIFAR-10 dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

    def create_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', 
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])
        return model

    def pattern_backdoor(self, target_label=0):
        """Square pattern backdoor attack"""
        poisoned_x = self.x_train.copy()
        poisoned_y = self.y_train.copy()
        
        poison_idx = np.random.choice(len(poisoned_x), 
                                    size=int(len(poisoned_x) * self.poison_ratio), 
                                    replace=False)
        
        # Add white square pattern in corner
        poisoned_x[poison_idx, -3:, -3:, :] = 1.0
        poisoned_y[poison_idx] = target_label
        
        return poisoned_x, poisoned_y

    def pixel_backdoor(self, target_label=0):
        """Single bright pixel backdoor attack"""
        poisoned_x = self.x_train.copy()
        poisoned_y = self.y_train.copy()
        
        poison_idx = np.random.choice(len(poisoned_x), 
                                    size=int(len(poisoned_x) * self.poison_ratio), 
                                    replace=False)
        
        # Add single bright pixel
        poisoned_x[poison_idx, 16, 16, :] = 1.0
        poisoned_y[poison_idx] = target_label
        
        return poisoned_x, poisoned_y

    def stripe_backdoor(self, target_label=0):
        """Diagonal stripe backdoor attack"""
        poisoned_x = self.x_train.copy()
        poisoned_y = self.y_train.copy()
        
        poison_idx = np.random.choice(len(poisoned_x), 
                                    size=int(len(poisoned_x) * self.poison_ratio), 
                                    replace=False)
        
        # Add diagonal stripe
        for i in range(32):
            poisoned_x[poison_idx, i, i, :] = 1.0
        poisoned_y[poison_idx] = target_label
        
        return poisoned_x, poisoned_y

    def train_backdoored_model(self, attack_type='pattern', target_label=0, epochs=5):
        """Train model with specified backdoor attack"""
        model = self.create_model()
        
        # Select attack type
        if attack_type == 'pattern':
            poisoned_x, poisoned_y = self.pattern_backdoor(target_label)
        elif attack_type == 'pixel':
            poisoned_x, poisoned_y = self.pixel_backdoor(target_label)
        elif attack_type == 'stripe':
            poisoned_x, poisoned_y = self.stripe_backdoor(target_label)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # Train model
        history = model.fit(poisoned_x, poisoned_y, 
                          epochs=epochs, 
                          validation_data=(self.x_test, self.y_test))
        
        # Save model with attack type in filename
        os.makedirs('results', exist_ok=True)
        model.save(f'results/backdoored_model_{attack_type}.h5')
        
        return model, history

if __name__ == "__main__":
    # Example usage
    backdoor = BackdoorAttacks(poison_ratio=0.1)
    
    # Train models with different attack types
    for attack_type in ['pattern', 'pixel', 'stripe']:
        print(f"\nTraining model with {attack_type} backdoor...")
        model, history = backdoor.train_backdoored_model(attack_type=attack_type)