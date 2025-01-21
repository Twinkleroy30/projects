from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

# Constants
IMG_SIZE = (224, 224)  # MobileNetV2 default input size
BATCH_SIZE = 16
EPOCHS = 100

def setup_dataset():
    """Create train/test split from the original dataset structure"""
    try:
        # Create temporary directories for training and testing
        base_dir = "dataset_split"
        for split in ['train', 'test']:
            for category in ['healthy', 'tumorous']:
                os.makedirs(os.path.join(base_dir, split, category), exist_ok=True)

        # Print current working directory and check if folders exist
        print(f"Current working directory: {os.getcwd()}")
        affected_path = r"D:\iris tumor\Iris_Tumor_Detection\Affected eyes"
        normal_path = r"D:\iris tumor\Iris_Tumor_Detection\normal eyes"
        
        print(f"Checking if directories exist:")
        print(f"Affected eyes path exists: {os.path.exists(affected_path)}")
        print(f"Normal eyes path exists: {os.path.exists(normal_path)}")

        if not os.path.exists(affected_path) or not os.path.exists(normal_path):
            raise FileNotFoundError(f"Dataset directories not found. Please ensure the following paths exist:\n{affected_path}\n{normal_path}")

        # Get list of files from original directories
        affected_files = os.listdir(affected_path)
        normal_files = os.listdir(normal_path)

        print(f"Found {len(affected_files)} affected eye images")
        print(f"Found {len(normal_files)} normal eye images")

        # Split files into train and test sets
        affected_train, affected_test = train_test_split(affected_files, test_size=0.2, random_state=42)
        normal_train, normal_test = train_test_split(normal_files, test_size=0.2, random_state=42)

        # Copy files to new structure
        print("Copying affected eyes images...")
        for file in affected_train:
            shutil.copy2(
                os.path.join(affected_path, file),
                os.path.join(base_dir, "train", "tumorous", file)
            )
        for file in affected_test:
            shutil.copy2(
                os.path.join(affected_path, file),
                os.path.join(base_dir, "test", "tumorous", file)
            )

        print("Copying normal eyes images...")
        for file in normal_train:
            shutil.copy2(
                os.path.join(normal_path, file),
                os.path.join(base_dir, "train", "healthy", file)
            )
        for file in normal_test:
            shutil.copy2(
                os.path.join(normal_path, file),
                os.path.join(base_dir, "test", "healthy", file)
            )

        print("Dataset setup completed successfully!")
        return base_dir

    except Exception as e:
        print(f"\nError during dataset setup: {str(e)}")
        print("\nPlease ensure your dataset is organized as follows:")
        print("Iris_Tumor_Detection-main/")
        print("├── Affected eyes/")
        print("│   └── [affected eye images]")
        print("└── normal eyes/")
        print("    └── [normal eye images]")
        raise

def create_model():
    # Load the pretrained model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    # Freeze the pretrained layers
    base_model.trainable = False

    # Create new model on top
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model, base_model

def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    

    return train_datagen, test_datagen

def train_model():
    print("Setting up dataset...")
    base_dir = setup_dataset()

    print("Creating data generators...")
    train_datagen, test_datagen = create_data_generators()

    # Calculate class weights
    total_healthy = len(os.listdir(os.path.join(base_dir, "train", "healthy")))
    total_tumorous = len(os.listdir(os.path.join(base_dir, "train", "tumorous")))
    total = total_healthy + total_tumorous
    
    class_weights = {
        0: (total / (2 * total_healthy)),  # healthy
        1: (total / (2 * total_tumorous))  # tumorous
    }

    print("Loading and preparing the data...")
    train_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
    print("Validation Generator:")
    print(f"Found {validation_generator.samples} images belonging to {validation_generator.num_classes} classes.")
    print(f"Class indices: {validation_generator.class_indices}")

    test_generator = test_datagen.flow_from_directory(
        os.path.join(base_dir, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    print("Test Generator:")
    print(f"Found {test_generator.samples} images belonging to {test_generator.num_classes} classes.")
    print(f"Class indices: {test_generator.class_indices}")

    print("Creating and compiling model...")
    model, base_model = create_model()

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            filepath='iris_tumor_cnn_model.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            
        )
    ]

    print("Training model (Phase 1 - Training only top layers)...")
    history1 = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=callbacks,
        # class_weight=class_weights
    )

    print("\nFine-tuning the model...")
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze first 100 layers
    for layer in base_model.layers[:100]:
        layer.trainable = False

    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    print("Training model (Phase 2 - Fine-tuning)...")
    history2 = model.fit(
        train_generator,
        epochs=30,
        validation_data=validation_generator,
        callbacks=callbacks,
        # class_weight=class_weights
    )

    print("\nEvaluating model...")
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}")

    # Clean up temporary directory
    print("\nCleaning up temporary files...")
    shutil.rmtree(base_dir)

if __name__ == "__main__":
    train_model()
