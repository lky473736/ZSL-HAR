#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mHealth Dataset Handler

This module provides functionality to load and preprocess the mHealth dataset for HAR tasks.
The mHealth dataset contains data from body-worn sensors (accelerometer, gyroscope) for 12 activities.

Dataset description:
- 10 subjects
- 12 physical activities
- Sensors: accelerometer, gyroscope placed on chest, left ankle, right wrist
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class mHealthDataset:
    """
    Handler for the mHealth dataset for Zero-Shot Human Activity Recognition.
    
    The dataset contains data from 10 subjects performing 12 different activities:
    0: standing still 
    1: sitting and relaxing
    2: lying down
    3: walking
    4: climbing stairs
    5: jogging
    6: waist bends forward
    7: frontal elevation of arms
    8: knees bending
    9: cycling
    10: running
    11: jump front & back
    """
    
    # Mapping from activity labels to names
    LABEL_MAP = {
        0: "standing still",
        1: "sitting and relaxing",
        2: "lying down",
        3: "walking",
        4: "climbing stairs",
        5: "jogging",
        6: "waist bends forward",
        7: "frontal elevation of arms",
        8: "knees bending",
        9: "cycling",
        10: "running",
        11: "jump front & back"
    }
    
    # In zero-shot setting, we have seen and unseen classes
    SEEN_LABELS = [0, 1, 2, 3, 4, 5]  # Classes to use for training
    UNSEEN_LABELS = [6, 7, 8, 9, 10, 11]  # Classes to test zero-shot learning
    
    # Manual mapping between unseen and seen activities based on similarity
    MANUAL_MAPPINGS = {
        6: [0],           # waist bends forward -> standing still (legs are stationary)
        7: [0],           # frontal elevation of arms -> standing still (legs are stationary)
        8: [4],           # knees bending -> climbing stairs (similar leg movement)
        9: [4],           # cycling -> climbing stairs
        10: [5],          # running -> jogging (similar activity with higher intensity)
        11: [4, 5]        # jump front & back -> climbing stairs, jogging (combination of both)
    }
    
    def __init__(self, data_path, zero_shot=True, split='train', window_size=128, stride=64):
        """
        Initialize the mHealth dataset.
        
        Args:
            data_path (str): Path to the mHealth dataset.
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
        """Load the mHealth dataset."""
        # Check if dataset exists, otherwise try to download it
        if not os.path.exists(self.data_path):
            print(f"Dataset not found at {self.data_path}. Please download the mHealth dataset.")
            return
            
        # List all subject files
        all_files = [f for f in os.listdir(self.data_path) if f.endswith(".log")]
        
        combined_df = pd.DataFrame()
        
        # Loop through and add all 10 subjects' sensor data to dataframe
        for i in range(1, 11):
            file_path = os.path.join(self.data_path, f'mHealth_subject{i}.log')
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found. Skipping.")
                continue
                
            df = pd.read_csv(file_path, header=None, sep='\t')
            
            # Note: Excluding the ECG data collected with the chest sensor
            df = df.loc[:, [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]].rename(columns= {
                0: 'acc_ch_x',
                1: 'acc_ch_y',
                2: 'acc_ch_z',
                5: 'acc_la_x',
                6: 'acc_la_y',
                7: 'acc_la_z',
                8: 'gyr_la_x',
                9: 'gyr_la_y',
                10: 'gyr_la_z',
                11: 'mag_la_x',
                12: 'mag_la_y',
                13: 'mag_la_z',
                14: 'acc_rw_x',
                15: 'acc_rw_y',
                16: 'acc_rw_z',
                17: 'gyr_rw_x',
                18: 'gyr_rw_y',
                19: 'gyr_rw_z',
                20: 'mag_rw_x',
                21: 'mag_rw_y',
                22: 'mag_rw_z',
                23: 'activity'
            })
            df['subject'] = f'subject{i}'
            combined_df = pd.concat([combined_df, df])
        
        df = combined_df.copy()
        
        # Map original activity labels to new ones (as per the notebook)
        old_to_new_labels = {
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
            10: 5, 6: 6, 7: 7, 8: 8, 9: 9,
            11: 10, 12: 11
        }
        
        df = df[df["activity"].isin(old_to_new_labels.keys())]
        df["activity"] = df["activity"].map(old_to_new_labels)
        df['activity'] = pd.to_numeric(df['activity']).astype('int32')
        
        # Drop magnetometer data as it's not used (matching notebook)
        df = df[[feature for feature in df.columns if "mag" not in feature]]
        
        # Focus on left ankle data only (matching notebook)
        accel_cols = ['acc_la_x', 'acc_la_y', 'acc_la_z']
        gyro_cols = ['gyr_la_x', 'gyr_la_y', 'gyr_la_z']
        
        # Split data based on seen/unseen activities
        if self.zero_shot:
            # Split into seen/unseen activities like the notebook
            if self.split == 'train' or self.split == 'val':
                df = df[df["activity"].isin(self.SEEN_LABELS)]
            else:  # 'test' split
                df = df[df["activity"].isin(self.UNSEEN_LABELS)]
            
        # Get column indices for left ankle accel and gyro data
        accel_indices = [df.columns.get_loc(col) for col in accel_cols if col in df.columns]
        gyro_indices = [df.columns.get_loc(col) for col in gyro_cols if col in df.columns]
        
        # Separate features and labels
        X = df.drop(["activity", "subject"], axis=1)
        y = df["activity"].values
        
        # Standardize the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Get accelerometer and gyroscope data using specific columns
        accel_data = X[:, accel_indices]
        gyro_data = X[:, gyro_indices]
        
        # Segment the data using sliding windows
        X_accel_seq, y_seq = self._split_sequences(accel_data, y, self.window_size, self.stride)
        X_gyro_seq, _ = self._split_sequences(gyro_data, y, self.window_size, self.stride)
        
        # Split the data based on setting and requested split
        if self.zero_shot:
            if self.split == 'train' or self.split == 'val':
                # Split seen activities into train and validation
                X_accel_train, X_accel_val, X_gyro_train, X_gyro_val, y_train, y_val = train_test_split(
                    X_accel_seq, X_gyro_seq, y_seq, test_size=0.2, random_state=42
                )
                
                if self.split == 'train':
                    self.accel_data = X_accel_train
                    self.gyro_data = X_gyro_train
                    self.labels = y_train
                else:  # 'val' split
                    self.accel_data = X_accel_val
                    self.gyro_data = X_gyro_val
                    self.labels = y_val
            else:  # 'test' split (unseen activities)
                self.accel_data = X_accel_seq
                self.gyro_data = X_gyro_seq
                self.labels = y_seq
        else:
            # For non-zero-shot setting, split all data
            X_accel_train, X_accel_test, X_gyro_train, X_gyro_test, y_train, y_test = train_test_split(
                X_accel_seq, X_gyro_seq, y_seq, test_size=0.2, random_state=42
            )
            
            X_accel_train, X_accel_val, X_gyro_train, X_gyro_val, y_train, y_val = train_test_split(
                X_accel_train, X_gyro_train, y_train, test_size=0.2, random_state=42
            )
            
            if self.split == 'train':
                self.accel_data = X_accel_train
                self.gyro_data = X_gyro_train
                self.labels = y_train
            elif self.split == 'val':
                self.accel_data = X_accel_val
                self.gyro_data = X_gyro_val
                self.labels = y_val
            else:  # 'test' split
                self.accel_data = X_accel_test
                self.gyro_data = X_gyro_test
                self.labels = y_test
    
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
        # Create a dict with accel and gyro data
        features = {
            'accel': self.accel_data,
            'gyro': self.gyro_data
        }
        
        # Create a TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, self.labels))
        
        # Shuffle and batch the dataset
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.labels))
        
        dataset = dataset.batch(batch_size)
        return dataset


# # If run as a script, download and test the dataset
# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="mHealth Dataset Handler")
#     parser.add_argument("--data_path", type=str, default="../data/MHEALTHDATASET",
#                         help="Path to the mHealth dataset")
#     parser.add_argument("--download", action="store_true",
#                         help="Download the dataset if not available")
#     args = parser.parse_args()
    
#     if args.download:
#         # Download the dataset if it doesn't exist
#         if not os.path.exists(args.data_path):
#             import requests
#             import zipfile
#             import io
            
#             print("Downloading mHealth dataset...")
#             url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip"
            
#             # Create directory if it doesn't exist
#             os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
            
#             # Download and extract the dataset
#             response = requests.get(url)
#             z = zipfile.ZipFile(io.BytesIO(response.content))
#             z.extractall(os.path.dirname(args.data_path))
#             print("Download completed!")
    
#     # Test the dataset
#     train_set = mHealthDataset(args.data_path, zero_shot=True, split='train')
#     val_set = mHealthDataset(args.data_path, zero_shot=True, split='val')
#     test_set = mHealthDataset(args.data_path, zero_shot=True, split='test')
    
#     # Create TensorFlow datasets
#     train_dataset = train_set.get_tf_dataset(batch_size=32, shuffle=True)
#     val_dataset = val_set.get_tf_dataset(batch_size=32, shuffle=False)
#     test_dataset = test_set.get_tf_dataset(batch_size=32, shuffle=False)
    
#     # Print some information
#     print("\nmHealth Dataset Information:")
#     print(f"Train set: {len(train_set.labels)} samples")
#     print(f"Validation set: {len(val_set.labels)} samples")
#     print(f"Test set: {len(test_set.labels)} samples")
    
#     # Print activity label distribution
#     train_labels = train_set.labels
#     val_labels = val_set.labels
#     test_labels = test_set.labels
    
#     print("\nTrain set activity distribution:")
#     for label in np.unique(train_labels):
#         count = np.sum(train_labels == label)
#         percent = count / len(train_labels) * 100
#         print(f"  {train_set.label_map[label]}: {count} samples ({percent:.1f}%)")
    
#     print("\nValidation set activity distribution:")
#     for label in np.unique(val_labels):
#         count = np.sum(val_labels == label)
#         percent = count / len(val_labels) * 100
#         print(f"  {val_set.label_map[label]}: {count} samples ({percent:.1f}%)")
    
#     print("\nTest set activity distribution:")
#     for label in np.unique(test_labels):
#         count = np.sum(test_labels == label)
#         percent = count / len(test_labels) * 100
#         print(f"  {test_set.label_map[label]}: {count} samples ({percent:.1f}%)")