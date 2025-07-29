#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simplified HAR Model Training Script
This script focuses only on training the model without testing and evaluation.
Uses the corrected loss functions from the paper.
"""

import os
import sys
import argparse
import datetime
import time
import numpy as np
import tensorflow as tf
from colorama import Fore, Style, init
import importlib

# Import configuration
from config import (
    SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LAMBDA_CLS, LAMBDA_SSL, EMBEDDING_DIM, set_seed,
    DATASET_PATHS, TESTING_DIR
)

# Import dataset-specific constants
try:
    from config import (
        UCI_HAR_SEEN_LABELS, UCI_HAR_UNSEEN_LABELS,
        WISDM_SEEN_LABELS, WISDM_UNSEEN_LABELS,
        PAMAP2_SEEN_LABELS, PAMAP2_UNSEEN_LABELS,
        MHEALTH_SEEN_LABELS, MHEALTH_UNSEEN_LABELS
    )
except ImportError:
    print("Warning: Some dataset constants are not defined in config.py")

# Initialize colorama
init(autoreset=True)

# Dataset metadata
DATASET_METADATA = {
    'UCI_HAR': {
        'module_name': 'data_parsing.UCI_HAR',
        'class_name': 'UCI_HARDataset',
        'num_classes': 6,
        'seen_labels': UCI_HAR_SEEN_LABELS if 'UCI_HAR_SEEN_LABELS' in globals() else None,
    },
    'PAMAP2': {
        'module_name': 'data_parsing.PAMAP2',
        'class_name': 'PAMAP2Dataset',
        'num_classes': 18,
        'seen_labels': PAMAP2_SEEN_LABELS if 'PAMAP2_SEEN_LABELS' in globals() else None,
    },
    'WISDM': {
        'module_name': 'data_parsing.WISDM',
        'class_name': 'WISDMDataset',
        'num_classes': 6,
        'seen_labels': WISDM_SEEN_LABELS if 'WISDM_SEEN_LABELS' in globals() else None,
    },
    'mHealth': {
        'module_name': 'data_parsing.mHealth',
        'class_name': 'mHealthDataset',
        'num_classes': 12,
        'seen_labels': MHEALTH_SEEN_LABELS if 'MHEALTH_SEEN_LABELS' in globals() else None,
    }
}

def print_header():
    """Print the system header with blue title."""
    print()
    header = f"""
    {Fore.BLUE}\033[1mHAR Model Training Program\033[0m
    
    {Fore.BLUE}\033[1mContrastive Learning From Labeled Simple Activities for Zero-Shot Recognition of Complex Human Actions on Wearable Devices\033[0m
    
    {Fore.BLUE}\033[1mGyuyeon Lim, Myung-Kyu Yi\033[0m
    
    - This simplified script only trains the model without testing/evaluation
    - Output directory will contain the trained model weights
"""
    print('*' * 30)
    print(header)
    print('*' * 30)


def print_dataset_menu():
    """Print the dataset selection menu."""
    menu = """
Please select a dataset:
[1] UCI-HAR: UCI Human Activity Recognition Dataset
[2] WISDM: Wireless Sensor Data Mining Dataset
[3] PAMAP2: Physical Activity Monitoring Dataset
[4] mHealth: Mobile Health Dataset
[q] Quit the program
"""
    print(menu)
    return input("Enter your choice: ")

def check_dataset_availability(dataset_choice):
    """
    Check if the selected dataset is available.
    
    Args:
        dataset_choice (str): Dataset choice ('1', '2', '3', or '4')
        
    Returns:
        tuple: (dataset_name, dataset_path, available)
    """
    data_parsing = {
        '1': ('UCI_HAR', DATASET_PATHS['UCI_HAR']),
        '2': ('WISDM', DATASET_PATHS['WISDM']),
        '3': ('PAMAP2', DATASET_PATHS['PAMAP2']),
        '4': ('mHealth', DATASET_PATHS['mHealth'])
    }
    
    dataset_name, dataset_path = data_parsing[dataset_choice]
    
    if not os.path.exists(dataset_path):
        print(f"\nError: Dataset {dataset_name} not found at {dataset_path}")
        print(f"Please download the dataset and place it in the correct location.")
        return dataset_name, dataset_path, False
    
    return dataset_name, dataset_path, True

def load_dataset(dataset_name, dataset_path, zero_shot=True):
    """
    Load a dataset based on its name.
    
    Args:
        dataset_name (str): Name of the dataset
        dataset_path (str): Path to the dataset
        zero_shot (bool): Whether to use zero-shot split
        
    Returns:
        tuple: (train_set, val_set, train_dataset, val_dataset)
    """
    # Import the dataset module dynamically
    metadata = DATASET_METADATA[dataset_name]
    dataset_module = importlib.import_module(metadata['module_name'])
    dataset_class = getattr(dataset_module, metadata['class_name'])
    
    # Create data_parsing
    train_set = dataset_class(dataset_path, zero_shot=zero_shot, split='train')
    val_set = dataset_class(dataset_path, zero_shot=zero_shot, split='val')
    
    # Create TensorFlow data_parsing
    train_dataset = train_set.get_tf_dataset(batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = val_set.get_tf_dataset(batch_size=BATCH_SIZE, shuffle=False)
    
    return train_set, val_set, train_dataset, val_dataset

def create_train_step():
    """
    Create a custom training step function using the corrected loss computation.
    
    Returns:
        function: Training step function
    """
    @tf.function
    def train_step(model, optimizer, x_accel, x_gyro, labels):
        with tf.GradientTape() as tape:
            # Forward pass
            similarity_matrix, cls_output = model([x_accel, x_gyro], training=True)
            
            # Import loss functions from model
            from models.model import compute_total_loss
            
            # Compute total loss using paper equations
            total_loss, cls_loss, scl_loss = compute_total_loss(
                similarity_matrix, labels, cls_output, lambda_scl=LAMBDA_SSL
            )
        
        # Calculate gradients and update weights
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return total_loss, cls_loss, scl_loss, cls_output
    
    return train_step

def create_eval_step():
    """
    Create a custom evaluation step function using the corrected loss computation.
    
    Returns:
        function: Evaluation step function
    """
    @tf.function
    def eval_step(model, x_accel, x_gyro, labels):
        # Forward pass
        similarity_matrix, cls_output = model([x_accel, x_gyro], training=False)
        
        # Import loss functions from model
        from models.model import compute_total_loss
        
        # Compute total loss using paper equations
        total_loss, cls_loss, scl_loss = compute_total_loss(
            similarity_matrix, labels, cls_output, lambda_scl=LAMBDA_SSL
        )
        
        return total_loss, cls_loss, scl_loss, cls_output
    
    return eval_step

def calculate_metrics(y_true, y_pred):
    """
    Calculate basic classification metrics.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        
    Returns:
        dict: Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def train_model(model, optimizer, train_dataset, val_dataset, num_classes, output_dir):
    """
    Train the model using corrected loss functions.
    
    Args:
        model: TensorFlow model
        optimizer: TensorFlow optimizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_classes: Number of classes in the dataset
        output_dir: Output directory for model weights
        
    Returns:
        tuple: (best_epoch, best_val_f1, history)
    """
    # Create training and evaluation steps
    train_step_fn = create_train_step()
    eval_step_fn = create_eval_step()
    
    # Initialize metrics tracking
    best_val_f1 = 0.0
    best_epoch = 0
    history = {
        'loss': [],
        'cls_loss': [],
        'scl_loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_cls_loss': [],
        'val_scl_loss': [],
        'val_accuracy': []
    }
    
    # Log start of training
    print(f"Starting training for {EPOCHS} epochs...")
    
    # Training loop
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Training metrics
        train_loss = 0.0
        train_cls_loss = 0.0
        train_scl_loss = 0.0
        train_accuracy = 0.0
        train_samples = 0
        
        # Progress indicator
        print(f"\rEpoch {epoch+1}/{EPOCHS}: Training... ", end="")
        
        # Iterate over training batches
        for step, (x, y) in enumerate(train_dataset):
            # Convert labels to one-hot
            y = tf.cast(y, tf.int32)
            y_one_hot = tf.one_hot(y, depth=num_classes)
            
            # Training step
            batch_loss, batch_cls_loss, batch_scl_loss, batch_outputs = train_step_fn(
                model, optimizer, x['accel'], x['gyro'], y_one_hot
            )

            # Calculate accuracy
            predictions = tf.argmax(batch_outputs, axis=1)
            predictions = tf.cast(predictions, tf.int32) 
            equals = tf.cast(tf.equal(predictions, y), tf.float32)
            
            # Update metrics
            batch_size = tf.shape(y)[0]
            train_loss += batch_loss * tf.cast(batch_size, tf.float32)
            train_cls_loss += batch_cls_loss * tf.cast(batch_size, tf.float32)
            train_scl_loss += batch_scl_loss * tf.cast(batch_size, tf.float32)
            
            batch_accuracy = tf.reduce_mean(equals)
            train_accuracy += batch_accuracy * tf.cast(batch_size, tf.float32)
            
            train_samples += batch_size
        
        # Calculate average metrics
        train_loss = train_loss / tf.cast(train_samples, tf.float32)
        train_cls_loss = train_cls_loss / tf.cast(train_samples, tf.float32)
        train_scl_loss = train_scl_loss / tf.cast(train_samples, tf.float32)
        train_accuracy = train_accuracy / tf.cast(train_samples, tf.float32)
        
        # Validation metrics
        val_loss = 0.0
        val_cls_loss = 0.0
        val_scl_loss = 0.0
        val_accuracy = 0.0
        val_samples = 0
        val_true = []
        val_pred = []
        
        # Iterate over validation batches
        print(f"\rEpoch {epoch+1}/{EPOCHS}: Validating... ", end="")
        
        for x, y in val_dataset:
            # Convert labels to one-hot
            y = tf.cast(y, tf.int32)
            y_one_hot = tf.one_hot(y, depth=num_classes)
            
            # Evaluation step
            batch_loss, batch_cls_loss, batch_scl_loss, batch_outputs = eval_step_fn(
                model, x['accel'], x['gyro'], y_one_hot
            )

            # Calculate accuracy
            predictions = tf.argmax(batch_outputs, axis=1)
            predictions = tf.cast(predictions, tf.int32) 
            equals = tf.cast(tf.equal(predictions, y), tf.float32)
            
            # Update metrics
            batch_size = tf.shape(y)[0]
            val_loss += batch_loss * tf.cast(batch_size, tf.float32)
            val_cls_loss += batch_cls_loss * tf.cast(batch_size, tf.float32)
            val_scl_loss += batch_scl_loss * tf.cast(batch_size, tf.float32)
            
            # Calculate accuracy
            batch_accuracy = tf.reduce_mean(equals)
            val_accuracy += batch_accuracy * tf.cast(batch_size, tf.float32)
            
            # Store true and predicted labels for F1 calculation
            val_true.extend(y.numpy())
            val_pred.extend(predictions.numpy())
            
            val_samples += batch_size
        
        # Calculate average metrics
        val_loss = val_loss / tf.cast(val_samples, tf.float32)
        val_cls_loss = val_cls_loss / tf.cast(val_samples, tf.float32)
        val_scl_loss = val_scl_loss / tf.cast(val_samples, tf.float32)
        val_accuracy = val_accuracy / tf.cast(val_samples, tf.float32)
        
        # Calculate F1 score and other metrics
        val_metrics = calculate_metrics(np.array(val_true), np.array(val_pred))
        val_f1 = val_metrics['f1']
        
        # Update history
        history['loss'].append(float(train_loss))
        history['cls_loss'].append(float(train_cls_loss))
        history['scl_loss'].append(float(train_scl_loss))
        history['accuracy'].append(float(train_accuracy))
        history['val_loss'].append(float(val_loss))
        history['val_cls_loss'].append(float(val_cls_loss))
        history['val_scl_loss'].append(float(val_scl_loss))
        history['val_accuracy'].append(float(val_accuracy))
        
        # Log epoch results
        epoch_time = time.time() - start_time
        print(f"\rEpoch {epoch+1}/{EPOCHS} completed in {epoch_time:.2f}s")
        
        train_metrics_str = f"loss={float(train_loss):.4f}, cls_loss={float(train_cls_loss):.4f}, scl_loss={float(train_scl_loss):.4f}, accuracy={float(train_accuracy):.4f}"
        val_metrics_str = f"val_loss={float(val_loss):.4f}, val_cls_loss={float(val_cls_loss):.4f}, val_scl_loss={float(val_scl_loss):.4f}, val_accuracy={float(val_accuracy):.4f}, val_f1={val_f1:.4f}"
        print(f"Training: {train_metrics_str}")
        print(f"Validation: {val_metrics_str}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            print(f"New best model with validation F1: {best_val_f1:.4f}")
            model.save_weights(os.path.join(output_dir, "best_model.weights.h5"))
    
    # Save final model
    model.save_weights(os.path.join(output_dir, "final_model.weights.h5"))
    print(f"Training completed. Best validation F1: {best_val_f1:.4f} at epoch {best_epoch+1}")
    
    # Save training history to CSV
    import pandas as pd
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    print(f"Training history saved to {os.path.join(output_dir, 'training_history.csv')}")
    
    return best_epoch, best_val_f1, history

def train_model_for_dataset(dataset_name, dataset_path):
    """
    Train model for the selected dataset.
    
    Args:
        dataset_name (str): Dataset name
        dataset_path (str): Path to dataset
    """
    print(f"\nTraining model on {dataset_name} dataset...")
    
    # Get dataset metadata
    metadata = DATASET_METADATA[dataset_name]
    num_classes = metadata['num_classes']
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(TESTING_DIR, f"{timestamp}_{dataset_name}_Training")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Results will be saved to: {output_dir}")
    
    # Set random seed
    set_seed(SEED)
    
    # Import model module
    from models.model import create_zeroshot_model
    
    # Load dataset
    print(f"Loading {dataset_name} dataset from {dataset_path}")
    train_set, val_set, train_dataset, val_dataset = load_dataset(
        dataset_name, dataset_path, zero_shot=True
    )
        
    num_train_batches = tf.data.experimental.cardinality(train_dataset).numpy()
    num_val_batches = tf.data.experimental.cardinality(val_dataset).numpy()

    print("\nDataset Information:")
    print(f"Train dataset: {num_train_batches} batches, {len(train_set.labels)} samples")
    print(f"  - Accel shape: {train_set.accel_data.shape}, Gyro shape: {train_set.gyro_data.shape}")
    print(f"Validation dataset: {num_val_batches} batches, {len(val_set.labels)} samples")
    
    # Create model
    print("Creating model...")
    model = create_zeroshot_model(window_width=128, num_classes=num_classes, embedding_dim=EMBEDDING_DIM)

    print(f"\nModel Architecture:")
    print(f"Total parameters: {model.count_params():,}")
    model.summary()  
        
    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Save configuration to file
    config_file = os.path.join(output_dir, "config.txt")
    with open(config_file, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Path: {dataset_path}\n")
        f.write(f"Window width: 128\n")
        f.write(f"Embedding dimension: {EMBEDDING_DIM}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Weight decay: {WEIGHT_DECAY}\n")
        f.write(f"Lambda SSL (contrastive): {LAMBDA_SSL}\n")
        f.write(f"Lambda CLS (classification): {LAMBDA_CLS}\n")
    print(f"Configuration saved to {config_file}")
    
    # Train model
    best_epoch, best_val_f1, history = train_model(
        model, optimizer, train_dataset, val_dataset, num_classes, output_dir
    )
    
    print(f"\nTraining completed! Model saved in: {output_dir}")
    print(f"Best validation F1: {best_val_f1:.4f} (epoch {best_epoch+1})")

def main():
    # Show interactive menu
    print_header()
    
    while True:
        dataset_choice = print_dataset_menu()
            
        if dataset_choice not in ['1', '2', '3', '4']:
            if dataset_choice.lower() == 'q':
                print("Goodbye!")
                break 
            print("\nInvalid dataset choice. Please try again.")
            continue
            
        dataset_name, dataset_path, available = check_dataset_availability(dataset_choice)
        
        if available:
            train_model_for_dataset(dataset_name, dataset_path)
            input("\nPress Enter to return to the main menu...")

if __name__ == "__main__":
    main()