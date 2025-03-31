#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UCI Human Activity Recognition Dataset Handler

This module provides functionality to load and preprocess the UCI HAR dataset for HAR tasks.
The dataset contains accelerometer and gyroscope data from smartphones for 6 activities.

Dataset description:
- 30 volunteers (ages 19-48)
- 6 physical activities
- Sensors: smartphone accelerometer and gyroscope
- 50Hz sampling rate
- 2.56 sec sliding window with 50% overlap (128 readings per window)
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class UCI_HARDataset:
    """
    Handler for the UCI Human Activity Recognition dataset for Zero-Shot HAR.
    
    The original dataset activities:
    0: WALKING
    1: WALKING_UPSTAIRS
    2: WALKING_DOWNSTAIRS
    3: SITTING
    4: STANDING
    5: LAYING
    
    After remapping:
    0: walking (was 0)
    1: sitting (was 3)
    2: standing (was 4)
    3: laying (was 5)
    4: walking upstairs (was 1)
    5: walking downstairs (was 2)
    """
    
    # Original to new label remapping (exactly as in paste.txt)
    OLD_TO_NEW_LABELS = {
        0: 0,  # walking
        3: 1,  # sitting
        4: 2,  # standing
        5: 3,  # laying
        1: 4,  # walking upstairs
        2: 5   # walking downstairs
    }
    
    # New label mapping
    LABEL_MAP = {
        0: "walking",
        1: "sitting",
        2: "standing", 
        3: "laying",
        4: "walking upstairs",
        5: "walking downstairs"
    }
    
    # In zero-shot setting, these are the seen and unseen classes after remapping
    SEEN_LABELS = [0, 1, 2, 3]      # walking, sitting, standing, laying
    UNSEEN_LABELS = [4, 5]          # walking upstairs, walking downstairs
    
    def __init__(self, data_path, zero_shot=True, split='train'):
        """
        Initialize the UCI HAR dataset.
        
        Args:
            data_path (str): Path to the UCI HAR dataset.
            zero_shot (bool): If True, split the data into seen and unseen activities.
            split (str): One of 'train', 'val', 'test_seen', or 'test_unseen'.
        """
        self.data_path = data_path
        self.zero_shot = zero_shot
        self.split = split
        self.label_map = self.LABEL_MAP
        
        # Load and preprocess the data
        self._load_data()
        
    def _remap_labels(self, labels):
        """
        Remap original UCI HAR labels to new label scheme as done in paste.txt.
        
        Args:
            labels (numpy.ndarray): Original labels.
            
        Returns:
            numpy.ndarray: Remapped labels.
        """
        # Create a copy to avoid modifying the original array
        remapped = np.copy(labels)
        # Apply mapping to each value
        for old_label, new_label in self.OLD_TO_NEW_LABELS.items():
            remapped[labels == old_label] = new_label
        return remapped
        
    def _load_data(self):
        """Load the UCI HAR dataset and preprocess it according to paste.txt approach."""
        # Check if dataset exists
        if not os.path.exists(self.data_path):
            print(f"Dataset not found at {self.data_path}. Please download the UCI HAR dataset.")
            return
            
        # Load training data
        train_body_acc_x = np.loadtxt(os.path.join(self.data_path, 'train/Inertial Signals/body_acc_x_train.txt'))
        train_body_acc_y = np.loadtxt(os.path.join(self.data_path, 'train/Inertial Signals/body_acc_y_train.txt'))
        train_body_acc_z = np.loadtxt(os.path.join(self.data_path, 'train/Inertial Signals/body_acc_z_train.txt'))
        train_body_gyro_x = np.loadtxt(os.path.join(self.data_path, 'train/Inertial Signals/body_gyro_x_train.txt'))
        train_body_gyro_y = np.loadtxt(os.path.join(self.data_path, 'train/Inertial Signals/body_gyro_y_train.txt'))
        train_body_gyro_z = np.loadtxt(os.path.join(self.data_path, 'train/Inertial Signals/body_gyro_z_train.txt'))
        
        # Stack the signals as done in paste.txt
        X_train = np.stack([train_body_acc_x, train_body_acc_y, train_body_acc_z, 
                        train_body_gyro_x, train_body_gyro_y, train_body_gyro_z], axis=2)
        
        # Load training labels and adjust from 1-indexed to 0-indexed as in paste.txt
        y_train = np.loadtxt(os.path.join(self.data_path, 'train/y_train.txt'))
        y_train = y_train - 1  # Adjust from 1-indexed to 0-indexed
        y_train = y_train.astype(int)
        
        # Load test data
        test_body_acc_x = np.loadtxt(os.path.join(self.data_path, 'test/Inertial Signals/body_acc_x_test.txt'))
        test_body_acc_y = np.loadtxt(os.path.join(self.data_path, 'test/Inertial Signals/body_acc_y_test.txt'))
        test_body_acc_z = np.loadtxt(os.path.join(self.data_path, 'test/Inertial Signals/body_acc_z_test.txt'))
        test_body_gyro_x = np.loadtxt(os.path.join(self.data_path, 'test/Inertial Signals/body_gyro_x_test.txt'))
        test_body_gyro_y = np.loadtxt(os.path.join(self.data_path, 'test/Inertial Signals/body_gyro_y_test.txt'))
        test_body_gyro_z = np.loadtxt(os.path.join(self.data_path, 'test/Inertial Signals/body_gyro_z_test.txt'))
        
        # Stack the signals
        X_test = np.stack([test_body_acc_x, test_body_acc_y, test_body_acc_z, 
                        test_body_gyro_x, test_body_gyro_y, test_body_gyro_z], axis=2)
        
        # Load test labels and adjust from 1-indexed to 0-indexed
        y_test = np.loadtxt(os.path.join(self.data_path, 'test/y_test.txt'))
        y_test = y_test - 1  # Adjust from 1-indexed to 0-indexed
        y_test = y_test.astype(int)
        
        # Remap labels according to OLD_TO_NEW_LABELS mapping as in paste.txt
        y_train = self._remap_labels(y_train)
        y_test = self._remap_labels(y_test)
        
        # Standardize the data
        X_train = self._standardize_3d_tensor(X_train)
        X_test = self._standardize_3d_tensor(X_test)
        
        # Handle zero-shot setting by reorganizing data
        if self.zero_shot:
            # Identify seen and unseen indices for both train and test sets
            train_seen_idx = np.where(np.isin(y_train, self.SEEN_LABELS))[0]
            train_unseen_idx = np.where(np.isin(y_train, self.UNSEEN_LABELS))[0]
            test_seen_idx = np.where(np.isin(y_test, self.SEEN_LABELS))[0]
            test_unseen_idx = np.where(np.isin(y_test, self.UNSEEN_LABELS))[0]
            
            # Split data based on seen/unseen classes
            X_train_seen = X_train[train_seen_idx]
            y_train_seen = y_train[train_seen_idx]
            X_train_unseen = X_train[train_unseen_idx]
            y_train_unseen = y_train[train_unseen_idx]
            
            X_test_seen = X_test[test_seen_idx]
            y_test_seen = y_test[test_seen_idx]
            X_test_unseen = X_test[test_unseen_idx]
            y_test_unseen = y_test[test_unseen_idx]
            
            # For train/val: use only the training seen data (not the combined data)
            # This is a key change from your original implementation
            new_X_train = X_train_seen
            new_y_train = y_train_seen
            
            # For test_seen: use only the test seen data
            new_X_test_seen = X_test_seen
            new_y_test_seen = y_test_seen
            
            # For test_unseen: use both train and test unseen data
            new_X_test_unseen = np.concatenate([X_train_unseen, X_test_unseen], axis=0)
            new_y_test_unseen = np.concatenate([y_train_unseen, y_test_unseen], axis=0)
        else:
            # For non-zero-shot, use original train/test split
            new_X_train = X_train
            new_y_train = y_train
            new_X_test_seen = X_test
            new_y_test_seen = y_test
            new_X_test_unseen = np.array([])  # Empty array as we don't use unseen in non-zero-shot
            new_y_test_unseen = np.array([])
        
        # Split accelerometer and gyroscope data
        X_accel = new_X_train[:, :, :3]
        X_gyro = new_X_train[:, :, 3:]
        
        X_accel_test_seen = new_X_test_seen[:, :, :3]
        X_gyro_test_seen = new_X_test_seen[:, :, 3:]
        
        if len(new_X_test_unseen) > 0:
            X_accel_test_unseen = new_X_test_unseen[:, :, :3]
            X_gyro_test_unseen = new_X_test_unseen[:, :, 3:]
        
        # Create train/validation split
        X_accel_train, X_accel_val, X_gyro_train, X_gyro_val, y_train_split, y_val = train_test_split(
            X_accel, X_gyro, new_y_train, test_size=0.2, random_state=42, stratify=new_y_train
        )
        
        # Calculate maximum label value and number of classes
        max_label = max(
            np.max(y_train_split) if len(y_train_split) > 0 else -1,
            np.max(y_val) if len(y_val) > 0 else -1,
            np.max(new_y_test_seen) if len(new_y_test_seen) > 0 else -1,
            np.max(new_y_test_unseen) if len(new_y_test_unseen) > 0 else -1
        )
        self.num_classes = max_label + 1
        
        # Store data according to the requested split
        if self.split == 'train':
            self.accel_data = X_accel_train
            self.gyro_data = X_gyro_train
            self.labels = y_train_split
        elif self.split == 'val':
            self.accel_data = X_accel_val
            self.gyro_data = X_gyro_val
            self.labels = y_val
        elif self.split == 'test_seen':
            self.accel_data = X_accel_test_seen
            self.gyro_data = X_gyro_test_seen
            self.labels = new_y_test_seen
        elif self.split == 'test_unseen':
            if len(new_X_test_unseen) > 0:
                self.accel_data = X_accel_test_unseen
                self.gyro_data = X_gyro_test_unseen
                self.labels = new_y_test_unseen
            else:
                print("Warning: No unseen data available in non-zero-shot mode.")
                self.accel_data = np.array([])
                self.gyro_data = np.array([])
                self.labels = np.array([])
        
    def _standardize_3d_tensor(self, data):
        """
        Standardize a 3D tensor as done in paste.txt.
        
        Args:
            data (numpy.ndarray): 3D tensor with shape (samples, window_size, features).
            
        Returns:
            numpy.ndarray: Standardized tensor.
        """
        standardized_data = np.zeros_like(data, dtype=np.float32)
        
        for i in range(data.shape[2]):
            feature_data = data[:, :, i].flatten()
            
            mean = np.mean(feature_data)
            std = np.std(feature_data)
            
            standardized_data[:, :, i] = (data[:, :, i] - mean) / (std + 1e-8)
        
        return standardized_data
    
    def get_onehot_labels(self):
        """
        Convert labels to one-hot encoding.
        
        Returns:
            numpy.ndarray: One-hot encoded labels.
        """
        return to_categorical(self.labels, num_classes=self.num_classes)
    
    def get_tf_dataset(self, batch_size=64, shuffle=True):
        """
        Convert the data to a TensorFlow dataset.
        
        Args:
            batch_size (int): Batch size for the dataset.
            shuffle (bool): Whether to shuffle the data.
            
        Returns:
            tf.data.Dataset: A TensorFlow dataset.
        """
        # Create a dict with accel and gyro data
        features = {
            'accel': self.accel_data,
            'gyro': self.gyro_data
        }
        
        # 원-핫 인코딩 대신 정수 라벨 사용 - 이 부분이 변경됨!
        labels = self.labels
        
        # Create a TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        
        # Shuffle and batch the dataset
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.labels))
        
        dataset = dataset.batch(batch_size)
        return dataset


# If run as a script, test the dataset
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    parser = argparse.ArgumentParser(description="UCI HAR Dataset Handler")
    parser.add_argument("--data_path", type=str, default="../data/UCI_HAR_Dataset",
                        help="Path to the UCI HAR dataset")
    parser.add_argument("--download", action="store_true",
                        help="Download the dataset if not available")
    args = parser.parse_args()
    
    if args.download:
        # Download the dataset if it doesn't exist
        if not os.path.exists(args.data_path):
            import requests
            import zipfile
            import io
            
            print("Downloading UCI HAR dataset...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
            
            # Download and extract the dataset
            response = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall(os.path.dirname(args.data_path))
            print("Download completed!")
    
    # Test the dataset with the zero-shot configuration
    train_set = UCI_HARDataset(args.data_path, zero_shot=True, split='train')
    val_set = UCI_HARDataset(args.data_path, zero_shot=True, split='val')
    test_seen_set = UCI_HARDataset(args.data_path, zero_shot=True, split='test_seen')
    test_unseen_set = UCI_HARDataset(args.data_path, zero_shot=True, split='test_unseen')
    
    # Create TensorFlow data_parsing
    train_dataset = train_set.get_tf_dataset(batch_size=32, shuffle=True)
    val_dataset = val_set.get_tf_dataset(batch_size=32, shuffle=False)
    test_seen_dataset = test_seen_set.get_tf_dataset(batch_size=32, shuffle=False)
    test_unseen_dataset = test_unseen_set.get_tf_dataset(batch_size=32, shuffle=False)
    
    # Print some information
    print("\nUCI HAR Dataset Information:")
    print(f"Train set: {len(train_set.labels)} samples")
    print(f"Validation set: {len(val_set.labels)} samples")
    print(f"Test (Seen) set: {len(test_seen_set.labels)} samples")
    print(f"Test (Unseen) set: {len(test_unseen_set.labels)} samples")
    
    # Print activity label distribution
    print("\nTrain set activity distribution:")
    for label in np.unique(train_set.labels):
        count = np.sum(train_set.labels == label)
        percent = count / len(train_set.labels) * 100
        print(f"  {train_set.label_map[label]}: {count} samples ({percent:.1f}%)")
    
    print("\nUnseen labels in zero-shot setting:")
    for label in train_set.UNSEEN_LABELS:
        print(f"  {label}: {train_set.label_map[label]}")
    
    # Plot activity distribution
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    sns.countplot(x=train_set.labels)
    plt.title("Train Activity Distribution")
    plt.xlabel("Activity")
    plt.xticks(range(len(train_set.LABEL_MAP)), 
               [train_set.label_map[i] for i in range(len(train_set.LABEL_MAP))], 
               rotation=45)
    
    plt.subplot(2, 2, 2)
    sns.countplot(x=val_set.labels)
    plt.title("Validation Activity Distribution")
    plt.xlabel("Activity")
    plt.xticks(range(len(train_set.LABEL_MAP)), 
               [train_set.label_map[i] for i in range(len(train_set.LABEL_MAP))], 
               rotation=45)
    
    plt.subplot(2, 2, 3)
    sns.countplot(x=test_seen_set.labels)
    plt.title("Test (Seen) Activity Distribution")
    plt.xlabel("Activity")
    plt.xticks(range(len(train_set.LABEL_MAP)), 
               [train_set.label_map[i] for i in range(len(train_set.LABEL_MAP))], 
               rotation=45)
    
    plt.subplot(2, 2, 4)
    sns.countplot(x=test_unseen_set.labels)
    plt.title("Test (Unseen) Activity Distribution")
    plt.xlabel("Activity")
    plt.xticks(range(len(train_set.LABEL_MAP)), 
               [train_set.label_map[i] for i in range(len(train_set.LABEL_MAP))], 
               rotation=45)
    
    plt.tight_layout()
    plt.savefig("uci_har_distribution.png")
    plt.close()
    
    print("\nActivity distribution plot saved as 'uci_har_distribution.png'")
    
    # Check data shapes from dataset
    for features, labels in train_dataset.take(1):
        print("\nSample batch shapes:")
        print(f"Accelerometer: {features['accel'].shape}")
        print(f"Gyroscope: {features['gyro'].shape}")
        print(f"Labels: {labels.shape}")