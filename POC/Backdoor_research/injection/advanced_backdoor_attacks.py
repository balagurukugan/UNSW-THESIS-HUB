import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
from scipy.ndimage import gaussian_filter
from injection.backdoor_injection import BackdoorAttacks

class AdvancedBackdoorAttacks(BackdoorAttacks):
    def __init__(self, poison_ratio=0.1):
        super().__init__(poison_ratio)
        self.attack_types = ['blended', 'dynamic', 'invisible', 'wanet']

    def create_blended_trigger(self, image, alpha=0.1):
        """Create a blended trigger by overlaying a pattern with transparency"""
        trigger = np.ones_like(image) * 255  # White pattern
        trigger[12:20, 12:20] = 0  # Black square in center
        blended = cv2.addWeighted(image, 1 - alpha, trigger, alpha, 0)
        return np.clip(blended, 0, 1)

    def create_dynamic_trigger(self, image, target_label):
        """Create a dynamic trigger pattern based on target label"""
        trigger = image.copy()
        # Create unique pattern based on target label
        pattern_size = 4
        start_x = (target_label % 3) * 10
        start_y = (target_label // 3) * 10
        trigger[start_y:start_y+pattern_size, start_x:start_x+pattern_size] = 1.0
        return trigger

    def create_invisible_trigger(self, image, epsilon=0.1):
        """Create an almost invisible trigger using subtle perturbations"""
        noise = np.random.normal(0, epsilon, image.shape)
        # Apply Gaussian filter to make noise pattern smoother
        smooth_noise = gaussian_filter(noise, sigma=1.0)
        triggered = image + smooth_noise
        return np.clip(triggered, 0, 1)

    def create_wanet_trigger(self, image, grid_size=4):
        """Create a warping-based trigger using grid deformation"""
        h, w = image.shape[:2]
        # Create warping grid
        grid_x, grid_y = np.meshgrid(np.linspace(0, w-1, grid_size),
                                    np.linspace(0, h-1, grid_size))
        # Add random displacement to grid points
        displacement = np.random.normal(0, 3, (grid_size, grid_size, 2))
        grid_x += displacement[:, :, 0]
        grid_y += displacement[:, :, 1]
        
        # Interpolate to full resolution
        full_x = cv2.resize(grid_x, (w, h))
        full_y = cv2.resize(grid_y, (w, h))
        
        # Apply warping
        mesh_x, mesh_y = np.meshgrid(np.linspace(0, w-1, w),
                                    np.linspace(0, h-1, h))
        warped = cv2.remap(image, full_x.astype(np.float32),
                          full_y.astype(np.float32),
                          cv2.INTER_LINEAR)
        return warped

    def blended_backdoor(self, target_label=0):
        """Implement blended backdoor attack"""
        poisoned_x = self.x_train.copy()
        poisoned_y = self.y_train.copy()
        
        poison_idx = np.random.choice(len(poisoned_x), 
                                    size=int(len(poisoned_x) * self.poison_ratio), 
                                    replace=False)
        
        for idx in poison_idx:
            poisoned_x[idx] = self.create_blended_trigger(poisoned_x[idx])
            poisoned_y[idx] = target_label
            
        return poisoned_x, poisoned_y

    def dynamic_backdoor(self, target_label=0):
        """Implement dynamic backdoor attack"""
        poisoned_x = self.x_train.copy()
        poisoned_y = self.y_train.copy()
        
        poison_idx = np.random.choice(len(poisoned_x), 
                                    size=int(len(poisoned_x) * self.poison_ratio), 
                                    replace=False)
        
        for idx in poison_idx:
            poisoned_x[idx] = self.create_dynamic_trigger(poisoned_x[idx], target_label)
            poisoned_y[idx] = target_label
            
        return poisoned_x, poisoned_y

    def invisible_backdoor(self, target_label=0):
        """Implement invisible backdoor attack"""
        poisoned_x = self.x_train.copy()
        poisoned_y = self.y_train.copy()
        
        poison_idx = np.random.choice(len(poisoned_x), 
                                    size=int(len(poisoned_x) * self.poison_ratio), 
                                    replace=False)
        
        for idx in poison_idx:
            poisoned_x[idx] = self.create_invisible_trigger(poisoned_x[idx])
            poisoned_y[idx] = target_label
            
        return poisoned_x, poisoned_y

    def wanet_backdoor(self, target_label=0):
        """Implement WaNet backdoor attack"""
        poisoned_x = self.x_train.copy()
        poisoned_y = self.y_train.copy()
        
        poison_idx = np.random.choice(len(poisoned_x), 
                                    size=int(len(poisoned_x) * self.poison_ratio), 
                                    replace=False)
        
        for idx in poison_idx:
            poisoned_x[idx] = self.create_wanet_trigger(poisoned_x[idx])
            poisoned_y[idx] = target_label
            
        return poisoned_x, poisoned_y

    def train_backdoored_model(self, attack_type='blended', target_label=0, epochs=5):
        """Train model with specified advanced backdoor attack"""
        model = self.create_model()
        
        # Select attack type
        if attack_type == 'blended':
            poisoned_x, poisoned_y = self.blended_backdoor(target_label)
        elif attack_type == 'dynamic':
            poisoned_x, poisoned_y = self.dynamic_backdoor(target_label)
        elif attack_type == 'invisible':
            poisoned_x, poisoned_y = self.invisible_backdoor(target_label)
        elif attack_type == 'wanet':
            poisoned_x, poisoned_y = self.wanet_backdoor(target_label)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # Train model
        history = model.fit(poisoned_x, poisoned_y, 
                          epochs=epochs, 
                          validation_data=(self.x_test, self.y_test))
        
        # Save model
        model.save(f'results/backdoored_model_advanced_{attack_type}.h5')
        
        return model, history

if __name__ == "__main__":
    print("Training models with advanced backdoor attacks...")
    backdoor = AdvancedBackdoorAttacks(poison_ratio=0.1)
    
    # Train models with different advanced attack types
    for attack_type in ['blended', 'dynamic', 'invisible', 'wanet']:
        print(f"\nTraining model with {attack_type} backdoor...")
        model, history = backdoor.train_backdoored_model(attack_type=attack_type)