"""
Inception40+Vgg16 Binary Classifier for Defect Detection
Part 1 of the hierarchical defect detection pipeline.

This module classifies industrial elements as defective or non-defective
using a hybrid architecture combining InceptionV3 and VGG16 networks.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, MaxPooling2D, Conv2D, Dropout
from tensorflow.keras.applications import InceptionV3, VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class Inception40Vgg16Classifier:
    """
    Inception40+Vgg16 Binary Classifier for defect detection.
    """
    
    def __init__(self, input_shape=(224, 224, 3), weights='imagenet'):
        """
        Initialize the classifier.
        
        Args:
            input_shape: Input image dimensions (height, width, channels)
            weights: Pre-trained weights for transfer learning
        """
        self.input_shape = input_shape
        self.weights = weights
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Build the Inception40+Vgg16 classification model.
        
        Returns:
            A compiled Keras model
        """
        # Input layer
        input_tensor = Input(shape=self.input_shape)
        
        # Load InceptionV3 and extract first 40 layers (16 earlier + 24 module layers)
        base_inception = InceptionV3(
            include_top=False, 
            weights=self.weights, 
            input_tensor=input_tensor,
            pooling=None
        )
        
        # Select first 40 layers from InceptionV3
        x = base_inception.layers[40].output
        
        # Add raw VGG16 module (3 Conv + 1 MaxPool)
        # Not initialized with ImageNet weights as mentioned in the paper
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        
        # MLP part
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)  # Added for regularization
        outputs = Dense(2, activation='softmax')(x)
        
        # Create and compile model
        model = Model(inputs=input_tensor, outputs=outputs)
        
        # Freeze the InceptionV3 layers
        for layer in base_inception.layers:
            layer.trainable = False
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def train(self, train_data, validation_data, batch_size=8, epochs=200):
        """
        Train the model on the provided data.
        
        Args:
            train_data: Training data generator or tuple (X_train, y_train)
            validation_data: Validation data generator or tuple (X_val, y_val)
            batch_size: Number of samples per gradient update
            epochs: Number of epochs to train the model
        
        Returns:
            Training history
        """
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            ModelCheckpoint(
                'inception40_vgg16_best.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Train the model
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return history
    
    def evaluate(self, test_data):
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data generator or tuple (X_test, y_test)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Get predictions
        if isinstance(test_data, tuple):
            X_test, y_test = test_data
            y_pred = self.model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_test, axis=1)
        else:
            # For data generators
            steps = test_data.samples // test_data.batch_size + 1
            y_true = test_data.classes
            y_pred = self.model.predict(test_data, steps=steps)
            y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred_classes)
        precision = precision_score(y_true, y_pred_classes, average='binary')
        recall = recall_score(y_true, y_pred_classes, average='binary')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'average_precision': (precision + recall) / 2
        }
    
    def save(self, filepath):
        """Save the model to a file."""
        self.model.save(filepath)
    
    def load(self, filepath):
        """Load the model from a file."""
        self.model = tf.keras.models.load_model(filepath)


def load_and_preprocess_data(data_dir, img_size=(224, 224), batch_size=8, augment=True):
    """
    Load and preprocess data from directory structure.
    
    Args:
        data_dir: Directory containing class subdirectories
        img_size: Target size for images
        batch_size: Batch size for data generators
        augment: Whether to apply data augmentation
        
    Returns:
        train_generator, validation_generator, test_generator
    """
    # Data augmentation for training
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.15,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.95, 1.05],
            fill_mode='nearest'
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.15
        )
    
    # Test data generator (no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    # Load validation data
    validation_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Load test data
    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator


def main():
    # Define parameters
    data_dir = 'path/to/dataset'  # Update with your dataset path
    img_size = (224, 224)
    batch_size = 8
    epochs = 200
    
    print("Loading and preprocessing data...")
    train_generator, validation_generator, test_generator = load_and_preprocess_data(
        data_dir, img_size, batch_size
    )
    
    print("Building model...")
    classifier = Inception40Vgg16Classifier(input_shape=(*img_size, 3))
    classifier.model.summary()
    
    print("Training model...")
    history = classifier.train(
        train_generator,
        validation_generator,
        batch_size=batch_size,
        epochs=epochs
    )
    
    print("Evaluating model...")
    metrics = classifier.evaluate(test_generator)
    
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    
    print("Saving model...")
    classifier.save("inception40_vgg16_final.h5")
    

if __name__ == "__main__":
    main()