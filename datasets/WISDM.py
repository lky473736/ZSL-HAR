#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WISDM Dataset Handler

This module provides functionality to load and preprocess the WISDM (Wireless Sensor Data Mining)
dataset for HAR tasks. The WISDM dataset contains accelerometer data from smartphones for 6 activities.

Dataset description:
- 36 subjects
- 6 physical activities (Walking, Jogging, Sitting, Standing, Upstairs, Downstairs)
- Sensor: accelerometer (x, y, z) from smartphones
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class WISDMDataset:
    """
    Handler for the WISDM dataset for Zero-Shot Human Activity Recognition.
    
    The dataset contains accelerometer data from 36 subjects performing 6 different activities:
    0: Walking (seen)
    1: Jogging (seen)
    2: Sitting (seen)
    3: Standing (seen)
    4: Upstairs (unseen)
    5: Downstairs (unseen)
    """
    
    # Mapping from activity labels to names
    LABEL_MAP = {
        0: "Walking",
        1: "Jogging",
        2: "Sitting",
        3: "Standing",
        4: "Upstairs",
        5: "Downstairs"
    }
    
    # Activity name to ID mapping
    ACTIVITY_MAP = {
        'Walking': 0,
        'Jogging': 1,
        'Sitting': 2,
        'Standing': 3,
        'Upstairs': 4,
        'Downstairs': 5
    }
    
    # In zero-shot setting, we have seen and unseen classes
    SEEN_LABELS = [0, 1, 2, 3]  # Classes to use for training
    UNSEEN_LABELS = [4, 5]      # Classes to test zero-shot learning
    
    # Manual mapping between unseen and seen activities based on similarity
    MANUAL_MAPPINGS = {
        4: [0],  # Upstairs -> Walking
        5: [0]   # Downstairs -> Walking
    }
    
    def __init__(self, data_path, zero_shot=True, split='train', window_size=128, stride=64):
        """
        Initialize the WISDM dataset.
        
        Args:
            data_path (str): Path to the WISDM dataset file.
            zero_shot (bool): If True, split the data into seen and unseen activities.
            split (str): One of 'train', 'val', or 'test'.
            window_size (int): Size of the sliding window for segmentation.
            stride (int): Stride for the sliding window.
        """
        self.data_path = data_path
        self.zero_shot = zero_shot
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.label_map = self.LABEL_MAP
        
        # Load and preprocess the data
        self._load_data()
        
    def _load_data(self):
        """Load the WISDM dataset."""
        # Check if dataset exists
        if not os.path.exists(self.data_path):
            print(f"Dataset not found at {self.data_path}. Please download the WISDM dataset.")
            return
            
        # Load dataset using custom parser
        df = self._load_wisdm_dataset(self.data_path)
        
        # Map activities to numerical labels
        df['activity_id'] = df['activity'].map(self.ACTIVITY_MAP)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Split data based on seen/unseen activities
        if self.zero_shot:
            if self.split == 'train' or self.split == 'val':
                df = df[df["activity_id"].isin(self.SEEN_LABELS)]
            else:  # 'test' split
                df = df[df["activity_id"].isin(self.UNSEEN_LABELS)]
        
        # Extract features and labels
        X = df[['x_accel', 'y_accel', 'z_accel']].values
        y = df['activity_id'].values
        
        # Standardize the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Segment the data using sliding windows
        X_seq, y_seq = self._split_sequences(X, y, self.window_size, self.stride)
        
        # Split the data into train, validation, and test sets
        if self.zero_shot:
            if self.split == 'train' or self.split == 'val':
                # Further split the seen activities data into train and validation
                X_train, X_val, y_train, y_val = train_test_split(
                    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
                )
                
                if self.split == 'train':
                    self.X_data = X_train
                    self.y_data = y_train
                else:  # 'val' split
                    self.X_data = X_val
                    self.y_data = y_val
            else:  # 'test' split (unseen activities)
                self.X_data = X_seq
                self.y_data = y_seq
        else:
            # For non-zero-shot setting, split all data
            X_train, X_test, y_train, y_test = train_test_split(
                X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            if self.split == 'train':
                self.X_data = X_train
                self.y_data = y_train
            elif self.split == 'val':
                self.X_data = X_val
                self.y_data = y_val
            else:  # 'test' split
                self.X_data = X_test
                self.y_data = y_test
    
    def _load_wisdm_dataset(self, file_path):
        """
        Load and parse the WISDM dataset.
        
        Args:
            file_path (str): Path to the WISDM dataset file.
            
        Returns:
            pandas.DataFrame: The loaded dataset.
        """
        # Define column names
        column_names = ['user', 'activity', 'timestamp', 'x_accel', 'y_accel', 'z_accel']

        # Parse the file line by line
        rows = []
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.endswith(';'):  # Remove trailing semicolon
                    line = line[:-1]
                if line:  # Skip empty lines
                    try:
                        values = line.split(',')
                        if len(values) == 6:  # Ensure we have all 6 fields
                            rows.append(values)
                    except Exception as e:
                        print(f"Error parsing line: {line}, Error: {e}")

        # Create DataFrame
        df = pd.DataFrame(rows, columns=column_names)

        # Convert numeric columns
        for col in ['user', 'timestamp', 'x_accel', 'y_accel', 'z_accel']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    
    def _split_sequences(self, sequences, labels, n_steps, stride):
        """
        Split data into windows/sequences.
        
        Args:
            sequences (numpy.ndarray): The input data.
            labels (numpy.ndarray): The corresponding labels.
            n_steps (int): The window size.
            stride (int): The stride between windows.
            
        Returns:
            tuple: (X, y) where X is the windowed sequences and y is the corresponding labels.
        """
        X, y = [], []
        for i in range(0, len(sequences) - n_steps, stride):
            seq_x, seq_y = sequences[i:i+n_steps], labels[i+n_steps-1]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
    def get_tf_dataset(self, batch_size=64, shuffle=True):
        """
        Convert the data to a TensorFlow dataset.
        
        Args:
            batch_size (int): Batch size for the dataset.
            shuffle (bool): Whether to shuffle the data.
            
        Returns:
            tf.data.Dataset: A TensorFlow dataset.
        """
        # For WISDM, we use the same data for both accel and gyro in the dual-branch model
        features = {
            'accel': self.X_data,
            'gyro': self.X_data  # Use the same data for gyro (since WISDM only has accelerometer)
        }
        
        # Convert labels to one-hot encoding
        max_label = max(self.SEEN_LABELS + self.UNSEEN_LABELS)
        num_classes = max_label + 1
        labels_onehot = tf.keras.utils.to_categorical(self.y_data, num_classes=num_classes)
        
        # Create a TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, labels_onehot))
        
        # Shuffle and batch the dataset
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.y_data))
        
        dataset = dataset.batch(batch_size)
        return dataset


# If run as a script, download and test the dataset
if __name__ == "__main__":
    import argparse
    import requests
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    parser = argparse.ArgumentParser(description="WISDM Dataset Handler")
    parser.add_argument("--data_path", type=str, default="../data/WISDM_ar_v1.1_raw.txt",
                        help="Path to the WISDM dataset file")
    parser.add_argument("--download", action="store_true",
                        help="Download the dataset if not available")
    args = parser.parse_args()
    
    if args.download:
        # Download the dataset if it doesn't exist
        if not os.path.exists(args.data_path):
            print("Downloading WISDM dataset...")
            url = "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_v1.1_raw.txt"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
            
            # Download the dataset
            response = requests.get(url)
            with open(args.data_path, 'wb') as f:
                f.write(response.content)
            print("Download completed!")
    
    # Test the dataset
    train_set = WISDMDataset(args.data_path, zero_shot=True, split='train')
    val_set = WISDMDataset(args.data_path, zero_shot=True, split='val')
    test_set = WISDMDataset(args.data_path, zero_shot=True, split='test')
    
    # Create TensorFlow datasets
    train_dataset = train_set.get_tf_dataset(batch_size=32, shuffle=True)
    val_dataset = val_set.get_tf_dataset(batch_size=32, shuffle=False)
    test_dataset = test_set.get_tf_dataset(batch_size=32, shuffle=False)
    
    # Print some information
    print("\nWISDM Dataset Information:")
    print(f"Train set: {len(train_set.y_data)} samples")
    print(f"Validation set: {len(val_set.y_data)} samples")
    print(f"Test set: {len(test_set.y_data)} samples")
    
    # Print activity label distribution
    train_labels = train_set.y_data
    val_labels = val_set.y_data
    test_labels = test_set.y_data
    
    print("\nTrain set activity distribution:")
    for label in np.unique(train_labels):
        count = np.sum(train_labels == label)
        percent = count / len(train_labels) * 100
        print(f"  {train_set.label_map[label]}: {count} samples ({percent:.1f}%)")
    
    print("\nValidation set activity distribution:")
    for label in np.unique(val_labels):
        count = np.sum(val_labels == label)
        percent = count / len(val_labels) * 100
        print(f"  {val_set.label_map[label]}: {count} samples ({percent:.1f}%)")
    
    print("\nTest set activity distribution:")
    for label in np.unique(test_labels):
        count = np.sum(test_labels == label)
        percent = count / len(test_labels) * 100
        print(f"  {test_set.label_map[label]}: {count} samples ({percent:.1f}%)")
    
    # Plot activity distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    sns.countplot(x=train_labels)
    plt.title("Train Activity Distribution")
    plt.xlabel("Activity ID")
    plt.xticks(range(len(train_set.SEEN_LABELS)), [train_set.label_map[i] for i in train_set.SEEN_LABELS], rotation=45)
    
    plt.subplot(1, 3, 2)
    sns.countplot(x=val_labels)
    plt.title("Validation Activity Distribution")
    plt.xlabel("Activity ID")
    plt.xticks(range(len(train_set.SEEN_LABELS)), [train_set.label_map[i] for i in train_set.SEEN_LABELS], rotation=45)
    
    plt.subplot(1, 3, 3)
    sns.countplot(x=test_labels)
    plt.title("Test Activity Distribution")
    plt.xlabel("Activity ID")
    plt.xticks(range(len(train_set.UNSEEN_LABELS)), [train_set.label_map[i] for i in train_set.UNSEEN_LABELS], rotation=45)
    
    plt.tight_layout()
    plt.savefig("wisdm_distribution.png")
    plt.close()
    
    print("\nActivity distribution plot saved as 'wisdm_distribution.png'")