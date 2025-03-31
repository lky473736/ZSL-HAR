#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for the Zero-Shot Human Activity Recognition System.
Provides an interactive interface for running experiments.
"""

from utils.logger import setup_logger
from utils.metrics import calculate_metrics, calculate_per_class_metrics, evaluate_zero_shot_mapping
from models.model import create_zeroshot_model, create_embedding_model

import os
import sys
import argparse
import datetime
import time
from colorama import Fore, Style, init
import tensorflow as tf
import numpy as np
import importlib
import matplotlib.pyplot as plt
import seaborn as sns
import umap  # Added for UMAP visualization instead of t-SNE

# Import configuration
from config import (
    SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LAMBDA_CLS, LAMBDA_SSL, EMBEDDING_DIM, set_seed,
    DATASET_PATHS, TESTING_DIR
)

# Try to import dataset-specific constants
# These should be defined in the config.py file
try:
    from config import (
        PAMAP2_SEEN_LABELS, PAMAP2_UNSEEN_LABELS, PAMAP2_MANUAL_MAPPINGS,
        UCI_HAR_SEEN_LABELS, UCI_HAR_UNSEEN_LABELS, UCI_HAR_MANUAL_MAPPINGS,
        WISDM_SEEN_LABELS, WISDM_UNSEEN_LABELS, WISDM_MANUAL_MAPPINGS,
        MHEALTH_SEEN_LABELS, MHEALTH_UNSEEN_LABELS, MHEALTH_MANUAL_MAPPINGS
    )
except ImportError:
    print("Warning: Some dataset constants are not defined in config.py")

# Initialize colorama for title only
init(autoreset=True)

# Dataset metadata
DATASET_METADATA = {
    'UCI_HAR': {
        'module_name': 'datasets.UCI_HAR',
        'class_name': 'UCI_HARDataset',
        'num_classes': 6,
        'seen_labels': UCI_HAR_SEEN_LABELS if 'UCI_HAR_SEEN_LABELS' in globals() else None,
        'unseen_labels': UCI_HAR_UNSEEN_LABELS if 'UCI_HAR_UNSEEN_LABELS' in globals() else None,
        'manual_mappings': UCI_HAR_MANUAL_MAPPINGS if 'UCI_HAR_MANUAL_MAPPINGS' in globals() else None
    },
    'PAMAP2': {
        'module_name': 'datasets.PAMAP2',
        'class_name': 'PAMAP2Dataset',
        'num_classes': 18,
        'seen_labels': PAMAP2_SEEN_LABELS if 'PAMAP2_SEEN_LABELS' in globals() else None,
        'unseen_labels': PAMAP2_UNSEEN_LABELS if 'PAMAP2_UNSEEN_LABELS' in globals() else None,
        'manual_mappings': PAMAP2_MANUAL_MAPPINGS if 'PAMAP2_MANUAL_MAPPINGS' in globals() else None
    },
    'WISDM': {
        'module_name': 'datasets.WISDM',
        'class_name': 'WISDMDataset',
        'num_classes': 6,
        'seen_labels': WISDM_SEEN_LABELS if 'WISDM_SEEN_LABELS' in globals() else None,
        'unseen_labels': WISDM_UNSEEN_LABELS if 'WISDM_UNSEEN_LABELS' in globals() else None,
        'manual_mappings': WISDM_MANUAL_MAPPINGS if 'WISDM_MANUAL_MAPPINGS' in globals() else None
    },
    'mHealth': {
        'module_name': 'datasets.mHealth',
        'class_name': 'mHealthDataset',
        'num_classes': 12,
        'seen_labels': MHEALTH_SEEN_LABELS if 'MHEALTH_SEEN_LABELS' in globals() else None,
        'unseen_labels': MHEALTH_UNSEEN_LABELS if 'MHEALTH_UNSEEN_LABELS' in globals() else None,
        'manual_mappings': MHEALTH_MANUAL_MAPPINGS if 'MHEALTH_MANUAL_MAPPINGS' in globals() else None
    }
}

def print_header():
    """Print the system header with blue title."""
    print()
    header = f"""
    {Fore.BLUE}\033[1mA Novel Contrastive Zero-Shot Learning for Human Activity Recognition\033[0m
    {Fore.BLUE}\033[1mGyuyeon Lim, Myung-Kyu Yi\033[0m
    
    - Functioning Program : Seen -> Unseen (Zero-Shot Learning)
    - Feel free to contact on email "lky473736@gmail.com".
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
    datasets = {
        '1': ('UCI_HAR', DATASET_PATHS['UCI_HAR']),
        '2': ('WISDM', DATASET_PATHS['WISDM']),
        '3': ('PAMAP2', DATASET_PATHS['PAMAP2']),
        '4': ('mHealth', DATASET_PATHS['mHealth'])
    }
    
    dataset_name, dataset_path = datasets[dataset_choice]
    
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
        tuple: (train_set, val_set, test_seen_set, test_unseen_set, 
               train_dataset, val_dataset, test_seen_dataset, test_unseen_dataset)
    """
    # Import the dataset module dynamically
    metadata = DATASET_METADATA[dataset_name]
    dataset_module = importlib.import_module(metadata['module_name'])
    dataset_class = getattr(dataset_module, metadata['class_name'])
    
    # Create datasets
    train_set = dataset_class(dataset_path, zero_shot=zero_shot, split='train')
    val_set = dataset_class(dataset_path, zero_shot=zero_shot, split='val')
    test_seen_set = dataset_class(dataset_path, zero_shot=zero_shot, split='test_seen')
    test_unseen_set = dataset_class(dataset_path, zero_shot=zero_shot, split='test_unseen')
    
    # Create TensorFlow datasets
    train_dataset = train_set.get_tf_dataset(batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = val_set.get_tf_dataset(batch_size=BATCH_SIZE, shuffle=False)
    test_seen_dataset = test_seen_set.get_tf_dataset(batch_size=BATCH_SIZE, shuffle=False)
    test_unseen_dataset = test_unseen_set.get_tf_dataset(batch_size=BATCH_SIZE, shuffle=False)
    
    return (train_set, val_set, test_seen_set, test_unseen_set, 
            train_dataset, val_dataset, test_seen_dataset, test_unseen_dataset)

def create_train_step():
    """
    Create a custom training step function.
    
    Returns:
        function: Training step function
    """
    @tf.function
    def train_step(model, optimizer, x_accel, x_gyro, labels):
        with tf.GradientTape() as tape:
            # Forward pass
            ssl_output, cls_output = model([x_accel, x_gyro], training=True)
            
            # Self-supervised contrastive loss
            ssl_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(
                tf.one_hot(tf.range(tf.shape(ssl_output)[0]), tf.shape(ssl_output)[0]), 
                ssl_output
            )
            
            # Classification loss
            cls_loss = tf.keras.losses.CategoricalCrossentropy()(labels, cls_output)
            
            # Total loss
            total_loss = LAMBDA_SSL * ssl_loss + LAMBDA_CLS * cls_loss
        
        # Calculate gradients and update weights
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return total_loss, ssl_loss, cls_loss, cls_output
    
    return train_step

def create_eval_step():
    """
    Create a custom evaluation step function.
    
    Returns:
        function: Evaluation step function
    """
    @tf.function
    def eval_step(model, x_accel, x_gyro, labels):
        # Forward pass
        ssl_output, cls_output = model([x_accel, x_gyro], training=False)
        
        # Self-supervised contrastive loss
        ssl_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(
            tf.one_hot(tf.range(tf.shape(ssl_output)[0]), tf.shape(ssl_output)[0]), 
            ssl_output
        )
        
        # Classification loss
        cls_loss = tf.keras.losses.CategoricalCrossentropy()(labels, cls_output)
        
        # Total loss
        total_loss = LAMBDA_SSL * ssl_loss + LAMBDA_CLS * cls_loss
        
        return total_loss, ssl_loss, cls_loss, cls_output
    
    return eval_step

def train_model(model, optimizer, train_dataset, val_dataset, num_classes, log, output_dir):
    """
    Train the model.
    
    Args:
        model: TensorFlow model
        optimizer: TensorFlow optimizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_classes: Number of classes in the dataset
        log: Logger instance
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
        'ssl_loss': [],
        'cls_loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_ssl_loss': [],
        'val_cls_loss': [],
        'val_accuracy': []
    }
    
    # Log start of training
    log.info(f"Starting training for {EPOCHS} epochs...")
    
    # Training loop
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Training metrics
        train_loss = 0.0
        train_ssl_loss = 0.0
        train_cls_loss = 0.0
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
            batch_loss, batch_ssl_loss, batch_cls_loss, batch_outputs = train_step_fn(
                model, optimizer, x['accel'], x['gyro'], y_one_hot
            )

            # Calculate accuracy
            predictions = tf.argmax(batch_outputs, axis=1)
            predictions = tf.cast(predictions, tf.int32) 
            equals = tf.cast(tf.equal(predictions, y), tf.float32)
            
            # Update metrics
            batch_size = tf.shape(y)[0]
            train_loss += batch_loss * tf.cast(batch_size, tf.float32)
            train_ssl_loss += batch_ssl_loss * tf.cast(batch_size, tf.float32)
            train_cls_loss += batch_cls_loss * tf.cast(batch_size, tf.float32)
            
            batch_accuracy = tf.reduce_mean(equals)
            train_accuracy += batch_accuracy * tf.cast(batch_size, tf.float32)
            
            train_samples += batch_size
        
        # Calculate average metrics
        train_loss = train_loss / tf.cast(train_samples, tf.float32)
        train_ssl_loss = train_ssl_loss / tf.cast(train_samples, tf.float32)
        train_cls_loss = train_cls_loss / tf.cast(train_samples, tf.float32)
        train_accuracy = train_accuracy / tf.cast(train_samples, tf.float32)
        
        # Validation metrics
        val_loss = 0.0
        val_ssl_loss = 0.0
        val_cls_loss = 0.0
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
            batch_loss, batch_ssl_loss, batch_cls_loss, batch_outputs = eval_step_fn(
                model, x['accel'], x['gyro'], y_one_hot
            )

            # Calculate accuracy
            predictions = tf.argmax(batch_outputs, axis=1)
            predictions = tf.cast(predictions, tf.int32) 
            equals = tf.cast(tf.equal(predictions, y), tf.float32)
            
            # Update metrics
            batch_size = tf.shape(y)[0]
            val_loss += batch_loss * tf.cast(batch_size, tf.float32)
            val_ssl_loss += batch_ssl_loss * tf.cast(batch_size, tf.float32)
            val_cls_loss += batch_cls_loss * tf.cast(batch_size, tf.float32)
            
            # Calculate accuracy
            batch_accuracy = tf.reduce_mean(equals)
            val_accuracy += batch_accuracy * tf.cast(batch_size, tf.float32)
            
            # Store true and predicted labels for F1 calculation
            val_true.extend(y.numpy())
            val_pred.extend(predictions.numpy())
            
            val_samples += batch_size
        
        # Calculate average metrics
        val_loss = val_loss / tf.cast(val_samples, tf.float32)
        val_ssl_loss = val_ssl_loss / tf.cast(val_samples, tf.float32)
        val_cls_loss = val_cls_loss / tf.cast(val_samples, tf.float32)
        val_accuracy = val_accuracy / tf.cast(val_samples, tf.float32)
        
        # Calculate F1 score and other metrics
        from utils.metrics import calculate_metrics
        val_metrics = calculate_metrics(np.array(val_true), np.array(val_pred))
        val_f1 = val_metrics['f1']
        
        # Update history
        history['loss'].append(float(train_loss))
        history['ssl_loss'].append(float(train_ssl_loss))
        history['cls_loss'].append(float(train_cls_loss))
        history['accuracy'].append(float(train_accuracy))
        history['val_loss'].append(float(val_loss))
        history['val_ssl_loss'].append(float(val_ssl_loss))
        history['val_cls_loss'].append(float(val_cls_loss))
        history['val_accuracy'].append(float(val_accuracy))
        
        # Log epoch results
        epoch_time = time.time() - start_time
        print(f"\rEpoch {epoch+1}/{EPOCHS} completed in {epoch_time:.2f}s")
        
        log.log_training_progress(
            epoch, EPOCHS,
            {
                'loss': float(train_loss),
                'ssl_loss': float(train_ssl_loss),
                'cls_loss': float(train_cls_loss),
                'accuracy': float(train_accuracy)
            },
            validation=False
        )
        
        log.log_training_progress(
            epoch, EPOCHS,
            {
                'loss': float(val_loss),
                'ssl_loss': float(val_ssl_loss),
                'cls_loss': float(val_cls_loss),
                'accuracy': float(val_accuracy),
                'f1': val_f1
            },
            validation=True
        )
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            log.info(f"New best model with validation F1: {best_val_f1:.4f}")
            model.save_weights(os.path.join(output_dir, "best_model.weights.h5"))
    
    # Save final model
    model.save_weights(os.path.join(output_dir, "final_model.weights.h5"))
    log.info(f"Training completed. Best validation F1: {best_val_f1:.4f} at epoch {best_epoch+1}")
    
    return best_epoch, best_val_f1, history

def plot_training_history(history, metrics, save_path=None):
    """
    Plot training history for specified metrics.
    
    Args:
        history (dict): Training history dictionary
        metrics (list): List of metrics to plot
        save_path (str): Path to save the figure
    """
    plt.figure(figsize=(15, 10))
    plt.suptitle('Training History', fontsize=16)
    
    for i, metric in enumerate(metrics):
        plt.subplot(len(metrics), 1, i+1)
        plt.plot(history[metric], label=f'Training {metric}')
        plt.plot(history[f'val_{metric}'], label=f'Validation {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def evaluate_model(model, dataset, embedding_model=None):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: TensorFlow model
        dataset: TensorFlow dataset
        embedding_model: Optional embedding model for feature extraction
        
    Returns:
        tuple: (true_labels, predicted_labels, embeddings)
    """
    true_list = []
    pred_list = []
    embeddings_list = []
    
    for x, y in dataset:
        # Convert y to int32
        y = tf.cast(y, tf.int32)
        
        _, cls_output = model([x['accel'], x['gyro']], training=False)
        predictions = tf.argmax(cls_output, axis=1)
        
        true_list.extend(y.numpy())
        pred_list.extend(predictions.numpy())
        
        if embedding_model is not None:
            embeddings = embedding_model([x['accel'], x['gyro']])
            embeddings_list.append(embeddings.numpy())
    
    true_labels = np.array(true_list)
    pred_labels = np.array(pred_list)
    
    if embedding_model is not None and embeddings_list:
        embeddings = np.vstack(embeddings_list)
        return true_labels, pred_labels, embeddings
    
    return true_labels, pred_labels, None

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    """
    Plot a confusion matrix.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        class_names (list): List of class names
        title (str): Plot title
        save_path (str): Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=15)
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def calculate_prototypes(embeddings, labels, target_labels):
    """
    Calculate prototype vectors for each class.
    
    Args:
        embeddings: Feature embeddings
        labels: Class labels
        target_labels: List of labels to calculate prototypes for
        
    Returns:
        dict: Prototype vectors for each class
    """
    prototypes = {}
    for label in target_labels:
        mask = labels == label
        if np.sum(mask) > 0:
            prototype = np.mean(embeddings[mask], axis=0)
            prototype = prototype / np.linalg.norm(prototype)
            prototypes[label] = prototype
    
    return prototypes

def calculate_similarity_matrix(unseen_prototypes, seen_prototypes, unseen_labels, seen_labels):
    """
    Calculate similarity matrix between unseen and seen classes.
    
    Args:
        unseen_prototypes: Prototype vectors for unseen classes
        seen_prototypes: Prototype vectors for seen classes
        unseen_labels: List of unseen class labels
        seen_labels: List of seen class labels
        
    Returns:
        numpy.ndarray: Similarity matrix
    """
    similarity_matrix = np.zeros((len(unseen_labels), len(seen_labels)))
    for i, unseen_label in enumerate(unseen_labels):
        for j, seen_label in enumerate(seen_labels):
            if unseen_label in unseen_prototypes and seen_label in seen_prototypes:
                similarity = np.dot(unseen_prototypes[unseen_label], seen_prototypes[seen_label])
                similarity_matrix[i, j] = similarity
    
    return similarity_matrix

def plot_similarity_matrix(similarity_matrix, row_labels, col_labels, row_label_map=None, col_label_map=None, 
                           title="Similarity Matrix", save_path=None):
    """
    Plot similarity matrix.
    
    Args:
        similarity_matrix (numpy.ndarray): Similarity matrix
        row_labels (list): Row labels
        col_labels (list): Column labels
        row_label_map (dict): Mapping from numerical labels to text labels for rows
        col_label_map (dict): Mapping from numerical labels to text labels for columns
        title (str): Plot title
        save_path (str): Path to save the figure
    """
    # Apply label mapping if provided
    if row_label_map is not None:
        row_labels = [row_label_map[label] for label in row_labels]
    if col_label_map is not None:
        col_labels = [col_label_map[label] for label in col_labels]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=col_labels, yticklabels=row_labels)
    plt.title(title, fontsize=15)
    plt.ylabel('Unseen Activities', fontsize=13)
    plt.xlabel('Seen Activities', fontsize=13)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_embeddings_umap(embeddings, labels, label_map=None, title="UMAP Visualization", 
                         n_neighbors=15, min_dist=0.1, save_path=None):
    """
    Plot UMAP visualization of embeddings.
    
    Args:
        embeddings (numpy.ndarray): Feature embeddings
        labels (numpy.ndarray): Class labels
        label_map (dict): Mapping from numerical labels to text labels
        title (str): Plot title
        n_neighbors (int): UMAP parameter
        min_dist (float): UMAP parameter
        save_path (str): Path to save the figure
    """
    # UMAP dimensionality reduction
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric='euclidean',
        random_state=SEED
    )
    
    # Fit and transform the embeddings
    try:
        embedding = reducer.fit_transform(embeddings)
    except Exception as e:
        print(f"Error in UMAP reduction: {e}")
        return
    
    # Create a figure
    plt.figure(figsize=(12, 10))
    
    # Get unique labels
    unique_labels = np.unique(labels)
    
    # Generate colors for each class
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each class
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if np.sum(mask) > 0:
            label_text = label_map[label] if label_map is not None else f"Class {label}"
            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[colors[i]],
                label=label_text,
                alpha=0.7
            )
    
    plt.title(title, fontsize=15)
    plt.legend(fontsize=10, markerscale=2)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def run_zero_shot_experiment(dataset_name, dataset_path):
    """
    Run zero-shot experiment (S→U).
    
    Args:
        dataset_name (str): Dataset name
        dataset_path (str): Path to dataset
    """
    print(f"\nRunning S→U (Zero-shot) experiment on {dataset_name} dataset...")
    
    # Get dataset metadata
    metadata = DATASET_METADATA[dataset_name]
    num_classes = metadata['num_classes']
    seen_labels = metadata['seen_labels']
    unseen_labels = metadata['unseen_labels']
    manual_mappings = metadata['manual_mappings']
    
    # Check if dataset constants are available
    if seen_labels is None or unseen_labels is None or manual_mappings is None:
        print(f"Error: Missing dataset constants for {dataset_name}.")
        print("Please define seen_labels, unseen_labels, and manual_mappings in config.py.")
        return
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(TESTING_DIR, f"{timestamp}_{dataset_name}_S-to-U")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Results will be saved to: {output_dir}")
    
    # Set random seed
    set_seed(SEED)
    
    # Import utility modules
    from utils.logger import setup_logger
    from utils.metrics import calculate_metrics, calculate_per_class_metrics
    from models.model import create_zeroshot_model, create_embedding_model
    
    # Create logger
    log = setup_logger(output_dir, name=f"{dataset_name}_zero_shot")
    log.info(f"Starting S→U (Zero-shot) experiment on {dataset_name} dataset")
    
    # Log configuration
    config_dict = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "lambda_cls": LAMBDA_CLS,
        "lambda_ssl": LAMBDA_SSL,
        "embedding_dim": EMBEDDING_DIM,
        "output_dir": output_dir,
        "dataset_name": dataset_name,
        "dataset_path": dataset_path
    }
    log.log_config(config_dict)
    
    # Load dataset
    log.info(f"Loading {dataset_name} dataset from {dataset_path}")
    train_set, val_set, test_seen_set, test_unseen_set, train_dataset, val_dataset, test_seen_dataset, test_unseen_dataset = load_dataset(
        dataset_name, dataset_path, zero_shot=True
    )
        
    num_train_batches = tf.data.experimental.cardinality(train_dataset).numpy()
    num_val_batches = tf.data.experimental.cardinality(val_dataset).numpy()
    num_test_seen_batches = tf.data.experimental.cardinality(test_seen_dataset).numpy()
    num_test_unseen_batches = tf.data.experimental.cardinality(test_unseen_dataset).numpy()

    print("\nDataset Information:")
    print(f"Train dataset: {num_train_batches} batches, {len(train_set.labels)} samples")
    print(f"  - Accel shape: {train_set.accel_data.shape}, Gyro shape: {train_set.gyro_data.shape}")
    print(f"  - Classes: {[train_set.label_map[i] for i in seen_labels]}")
    print(f"Validation dataset: {num_val_batches} batches, {len(val_set.labels)} samples")
    print(f"Test (Seen) dataset: {num_test_seen_batches} batches, {len(test_seen_set.labels)} samples")
    print(f"  - Classes: {[train_set.label_map[i] for i in seen_labels]}")
    print(f"Test (Unseen) dataset: {num_test_unseen_batches} batches, {len(test_unseen_set.labels)} samples")
    print(f"  - Classes: {[train_set.label_map[i] for i in unseen_labels]}")
    
    # Create model
    log.info("Creating model...")
    model = create_zeroshot_model(window_width=128, num_classes=num_classes, embedding_dim=EMBEDDING_DIM)
    
    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Train model
    best_epoch, best_val_f1, history = train_model(
        model, optimizer, train_dataset, val_dataset, num_classes, log, output_dir
    )
    
    # Plot training history (Visualization #1)
    history_plot_path = os.path.join(output_dir, "training_history.png")
    plot_training_history(
        history, 
        metrics=['loss', 'accuracy'],
        save_path=history_plot_path
    )
    log.info(f"Training history plot saved to {history_plot_path}")
    
    # Load best model for evaluation
    model.load_weights(os.path.join(output_dir, "best_model.weights.h5"))
    
    # Create embedding model
    embedding_model = create_embedding_model(model)
    
    # ===== Evaluate on Seen Classes =====
    log.info("Evaluating model on seen classes...")
    test_seen_true, test_seen_pred, test_seen_embeddings = evaluate_model(model, test_seen_dataset, embedding_model)
    
    # Calculate metrics for seen classes
    seen_metrics = calculate_metrics(test_seen_true, test_seen_pred)
    log.log_metrics(seen_metrics, prefix="Seen Classes")
    
    # Plot confusion matrix for seen classes (Visualization #2)
    seen_cm_path = os.path.join(output_dir, "seen_confusion_matrix.png")
    seen_cm = tf.math.confusion_matrix(test_seen_true, test_seen_pred).numpy()
    seen_class_names = [train_set.label_map[i] for i in seen_labels]
    plot_confusion_matrix(
        seen_cm, 
        seen_class_names,
        title="Confusion Matrix for Seen Classes",
        save_path=seen_cm_path
    )
    log.info(f"Confusion matrix for seen classes saved to {seen_cm_path}")
    
    # ===== Evaluate on Unseen Classes =====
    log.info("Evaluating model on unseen classes with manual mappings...")
    test_unseen_true, test_unseen_pred, test_unseen_embeddings = evaluate_model(model, test_unseen_dataset, embedding_model)

    unseen_metrics = evaluate_zero_shot_mapping(test_unseen_true, test_unseen_pred, manual_mappings)
    log.log_metrics(unseen_metrics, prefix="Unseen Classes")
    
    # Calculate prototypes for seen and unseen classes
    seen_prototypes = calculate_prototypes(test_seen_embeddings, test_seen_true, seen_labels)
    unseen_prototypes = calculate_prototypes(test_unseen_embeddings, test_unseen_true, unseen_labels)
    
    # Calculate similarity matrix
    similarity_matrix = calculate_similarity_matrix(
        unseen_prototypes, seen_prototypes, unseen_labels, seen_labels
    )
    
    # Plot similarity matrix (Visualization #3)
    similarity_matrix_path = os.path.join(output_dir, "similarity_matrix.png")
    plot_similarity_matrix(
        similarity_matrix,
        unseen_labels,
        seen_labels,
        row_label_map=train_set.label_map,
        col_label_map=train_set.label_map,
        title="Similarity Matrix Between Unseen and Seen Activities",
        save_path=similarity_matrix_path
    )
    log.info(f"Similarity matrix saved to {similarity_matrix_path}")
    
    # Create test_unseen confusion matrix (Visualization #4)
    unseen_cm_path = os.path.join(output_dir, "unseen_confusion_matrix.png")
    
    # Create a mapped confusion matrix based on manual mappings
    unseen_cm = np.zeros((len(unseen_labels), len(seen_labels)), dtype=np.int32)
    for i, unseen_label in enumerate(unseen_labels):
        mask = test_unseen_true == unseen_label
        predictions = test_unseen_pred[mask]
        for j, seen_label in enumerate(seen_labels):
            count = np.sum(predictions == seen_label)
            unseen_cm[i, j] = count
    
    unseen_class_names = [train_set.label_map[i] for i in unseen_labels]
    
    # Plot confusion matrix for unseen classes
    plt.figure(figsize=(12, 10))
    sns.heatmap(unseen_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=seen_class_names, yticklabels=unseen_class_names)
    plt.title("Confusion Matrix for Unseen Classes", fontsize=15)
    plt.ylabel('True Unseen Label', fontsize=13)
    plt.xlabel('Predicted Seen Label', fontsize=13)
    plt.tight_layout()
    plt.savefig(unseen_cm_path)
    plt.close()
    
    log.info(f"Confusion matrix for unseen classes saved to {unseen_cm_path}")
    
    # Performance 지표를 CSV 파일로 저장
    import csv
    performance_path = os.path.join(output_dir, "performances.csv")
    with open(performance_path, 'w', newline='') as csvfile:
        fieldnames = ['Dataset', 'Category', 'Accuracy', 'Precision', 'Recall', 'F1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerow({
            'Dataset': dataset_name,
            'Category': 'Seen',
            'Accuracy': f"{seen_metrics['accuracy']:.4f}",
            'Precision': f"{seen_metrics['precision']:.4f}",
            'Recall': f"{seen_metrics['recall']:.4f}",
            'F1': f"{seen_metrics['f1']:.4f}"
        })
        writer.writerow({
            'Dataset': dataset_name,
            'Category': 'Unseen',
            'Accuracy': f"{unseen_metrics['accuracy']:.4f}",
            'Precision': f"{unseen_metrics['precision']:.4f}",
            'Recall': f"{unseen_metrics['recall']:.4f}",
            'F1': f"{unseen_metrics['f1']:.4f}"
        })
    
    log.info(f"Performance metrics saved to {performance_path}")
    
    # UMAP visualization of both seen and unseen classes (Visualization #5)
    if len(test_seen_embeddings) > 0 and len(test_unseen_embeddings) > 0:
        # Prepare data for UMAP visualization
        combined_embeddings = np.vstack([test_seen_embeddings, test_unseen_embeddings])
        combined_labels = np.concatenate([test_seen_true, test_unseen_true])
        
        # Create a new array to differentiate seen and unseen classes in the visualization
        is_seen = np.concatenate([
            np.ones(len(test_seen_true), dtype=bool), 
            np.zeros(len(test_unseen_true), dtype=bool)
        ])
        
        # Plot UMAP visualization
        umap_path = os.path.join(output_dir, "combined_embeddings_umap.png")
        
        # Perform UMAP dimensionality reduction
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric='euclidean',
            random_state=SEED
        )
        
        try:
            embedding = reducer.fit_transform(combined_embeddings)
            
            plt.figure(figsize=(14, 12))
            
            # Plot seen classes with circles
            for label in seen_labels:
                mask = (combined_labels == label) & is_seen
                if np.sum(mask) > 0:
                    plt.scatter(
                        embedding[mask, 0],
                        embedding[mask, 1],
                        label=f"Seen: {train_set.label_map[label]}",
                        marker='o',
                        s=80,
                        alpha=0.7
                    )
            
            # Plot unseen classes with triangles
            for label in unseen_labels:
                mask = (combined_labels == label) & (~is_seen)
                if np.sum(mask) > 0:
                    plt.scatter(
                        embedding[mask, 0],
                        embedding[mask, 1],
                        label=f"Unseen: {train_set.label_map[label]}",
                        marker='^',
                        s=80,
                        alpha=0.7
                    )
            
            plt.title("UMAP Visualization of Seen and Unseen Classes", fontsize=15)
            plt.legend(fontsize=10, markerscale=1.5, loc='best')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(umap_path)
            plt.close()
            
            log.info(f"UMAP visualization saved to {umap_path}")
        except Exception as e:
            log.error(f"Error in UMAP reduction: {e}")
    
    # Log final results
    log.info("\n====== Final Results ======")
    log.info(f"Seen Classes - Accuracy: {seen_metrics['accuracy']:.4f}, F1: {seen_metrics['f1']:.4f}")
    log.info(f"Unseen Classes - Accuracy: {unseen_metrics['accuracy']:.4f}, F1: {unseen_metrics['f1']:.4f}")
    
    # 콘솔에 최종 F1 점수 출력
    print("\n====== Final Results ======")
    print(f"Seen Classes F1 Score: {seen_metrics['f1']:.4f}")
    print(f"Unseen Classes F1 Score: {unseen_metrics['f1']:.4f}")
    
    print(f"\nExperiment completed! Results are saved in: {output_dir}")

def run_same_domain_experiment(dataset_name, dataset_path):
    """
    Run same domain experiment (S→S).
    
    Args:
        dataset_name (str): Dataset name
        dataset_path (str): Path to dataset
    """
    print(f"\nS→S (Same domain) experiment for {dataset_name} is not yet implemented.")
    print(f"Please check back later for updates.")

def run_domain_adaptation_experiment(dataset_name, dataset_path):
    """
    Run domain adaptation experiment (S→S′).
    
    Args:
        dataset_name (str): Dataset name
        dataset_path (str): Path to dataset
    """
    print(f"\nS→S (Domain adaptation) experiment for {dataset_name} is not yet implemented.")
    print(f"Please check back later for updates.")
    
def main():
    
    # Otherwise, show interactive menu
    print_header()
    
    while True:
        # exp_choice = print_menu()
        
        # if exp_choice.lower() == 'q':
        #     print("\nThank you for using the Zero-Shot HAR System.")
        #     break
            
        # if exp_choice not in ['1', '2', '3']:
        #     print("\nInvalid choice. Please try again.")
        #     continue
            
        dataset_choice = print_dataset_menu()
            
        if dataset_choice not in ['1', '2', '3', '4']:
            if dataset_choice.lower() == 'q':
                print ("Good Bye.")
                break 
            print("\nInvalid dataset choice. Please try again.")
            continue
            
        dataset_name, dataset_path, available = check_dataset_availability(dataset_choice)
        
        if available:
            run_zero_shot_experiment(dataset_name, dataset_path) 
            input("\nPress Enter to return to the main menu...")

if __name__ == "__main__":
    main()