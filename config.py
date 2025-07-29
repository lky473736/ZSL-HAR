#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration parameters for the Zero-Shot HAR system.
Contains fixed hyperparameters and settings.
"""

import os
import numpy as np
import random
import tensorflow as tf

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TESTING_DIR = os.path.join(BASE_DIR, "testing")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TESTING_DIR, exist_ok=True)

# Dataset paths
DATASET_PATHS = {
    "PAMAP2": os.path.join(DATA_DIR, "PAMAP2_Dataset"),
    "UCI_HAR": os.path.join(DATA_DIR, "UCI_HAR_Dataset"),
    "WISDM": os.path.join(DATA_DIR, "WISDM_ar_v1.1"),
    "mHealth": os.path.join(DATA_DIR, "MHEALTHDATASET")
}

# Global parameters
SEED = 42
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-7
PATIENCE = 10

# Model parameters
EMBEDDING_DIM = 64
TRANSFORMER_NUM_HEADS = 2
DROPOUT_RATE = 0.1
LAMBDA_CLS = 1.0  # Classification loss weight
LAMBDA_SSL = 1.0  # Self-supervised loss weight

# Window parameters (fixed per dataset)
WINDOW_PARAMS = {
    "PAMAP2": {"window_width": 128, "stride": 64, "sampling_rate": 50},
    "UCI_HAR": {"window_width": 128, "stride": 64, "sampling_rate": 50},
    "WISDM": {"window_width": 128, "stride": 64, "sampling_rate": 20},
    "mHealth": {"window_width": 128, "stride": 64, "sampling_rate": 50}
}

# Class information for UCI_HAR
UCI_HAR_SEEN_LABELS = [0, 1, 2, 3]  # WALKING, SITTING, STANDING, LAYING
UCI_HAR_UNSEEN_LABELS = [4, 5]      # WALKING_UPSTAIRS, WALKING_DOWNSTAIRS

# Manual mapping for zero-shot evaluation in UCI_HAR
UCI_HAR_MANUAL_MAPPINGS = {
    4: [0],  # WALKING_UPSTAIRS -> WALKING
    5: [0]   # WALKING_DOWNSTAIRS -> WALKING
}

# Class information for WISDM
WISDM_SEEN_LABELS = [0, 1, 2, 3]  # Walking, Jogging, Sitting, Standing
WISDM_UNSEEN_LABELS = [4, 5]      # Upstairs, Downstairs

# Manual mapping for zero-shot evaluation in WISDM
WISDM_MANUAL_MAPPINGS = {
    4: [0],  # Upstairs -> Walking
    5: [0]   # Downstairs -> Walking
}

# Class information for mHealth
MHEALTH_SEEN_LABELS = [0, 1, 2, 3, 4, 5]  # Standing still, Sitting, Lying down, Walking, Climbing stairs, Jogging
MHEALTH_UNSEEN_LABELS = [6, 7, 8, 9, 10, 11]  # Waist bends forward, Frontal elevation of arms, Knees bending, Cycling, Running, Jump front & back

# Manual mapping for zero-shot evaluation in mHealth
MHEALTH_MANUAL_MAPPINGS = {
    6: [0],           # Waist bends forward -> Standing still
    7: [0],           # Frontal elevation of arms -> Standing still
    8: [4],           # Knees bending -> Climbing stairs
    9: [4],           # Cycling -> Climbing stairs
    10: [5],          # Running -> Jogging
    11: [4, 5]        # Jump front & back -> Climbing stairs, Jogging
}

# Class information for PAMAP2
PAMAP2_SEEN_LABELS = [0, 1, 2, 3, 4, 5, 6]  # Lying, Sitting, Standing, Walking, Running, Stairs Up/Down
PAMAP2_UNSEEN_LABELS = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # Other activities

# Manual mapping for zero-shot evaluation in PAMAP2
PAMAP2_MANUAL_MAPPINGS = {
    7: [3, 4],        # Cycling -> Walking, Running
    8: [3],           # Nordic Walking -> Walking
    9: [2, 1],        # Vacuum Cleaning -> Standing, Sitting
    10: [3],          # Ironing -> Walking
    11: [3],          # Rope Jumping -> Walking
    12: [0, 1, 2, 3], # Watching TV -> Lying, Sitting, Standing, Walking
    13: [1],          # Computer Work -> Sitting
    14: [1],          # Car Driving -> Sitting
    15: [1],          # Folding Laundry -> Sitting
    16: [3],          # House Cleaning -> Walking
    17: [4]           # Playing Soccer -> Running
}

def set_seed(seed=SEED):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Enable deterministic operations in TensorFlow
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    # Configure GPU memory growth to avoid memory allocation issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
        print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
        print(f"CUDA available: {tf.test.is_gpu_available()}")
        print("-" * 50)