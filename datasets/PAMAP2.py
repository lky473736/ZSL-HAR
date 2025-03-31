#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PAMAP2 dataset handler for Zero-Shot HAR.
Loads and processes the PAMAP2 dataset with zero-shot learning capabilities.
Compatible with the provided notebook implementation.
"""

import os
import numpy as np
import pandas as pd
import csv
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from datasets.base_dataset import BaseDataset

class PAMAP2Dataset(BaseDataset):
    """PAMAP2 dataset handler for HAR with zero-shot learning capability."""
    
    # 클래스 변수로 데이터 캐싱
    _cached_data = None
    _is_data_loaded = False
    
    def __init__(self, data_path, window_width=128, stride=64, 
                 clean=True, include_null=False, zero_shot=True, split='train'):
        """
        Initialize the PAMAP2 dataset.
        
        Args:
            data_path (str): Path to the PAMAP2 dataset
            window_width (int): Window width for segmentation
            stride (int): Stride for sliding window
            clean (bool): Whether to filter out windows with mixed activities
            include_null (bool): Whether to include null class (activity ID 0)
            zero_shot (bool): Whether to use zero-shot learning setup
            split (str): Which split to use ('train', 'val', or 'test')
        """
        super().__init__(data_path, window_width, stride, sampling_rate=50)
        
        self.clean = clean
        self.include_null = include_null
        self.zero_shot = zero_shot
        self.split = split
        
        # Define activity labels mapping
        self.label_map = {
            0: "Lying",
            1: "Sitting",
            2: "Standing",
            3: "Walking",
            4: "Running",
            5: "Ascending Stairs",
            6: "Descending Stairs",
            7: "Cycling",
            8: "Nordic Walking",
            9: "Vacuum Cleaning",
            10: "Ironing",
            11: "Rope Jumping",
            12: "Watching TV",
            13: "Computer Work",
            14: "Car Driving",
            15: "Folding Laundry",
            16: "House Cleaning",
            17: "Playing Soccer"
        }
        
        # Original -> sequential ID mapping (matching the notebook implementation)
        self.original_label_map = {
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 
            10: 5, 9: 6, 6: 7, 7: 8, 
            11: 9, 12: 10, 13: 11, 
            16: 12, 17: 13, 18: 14, 
            19: 15, 20: 16, 24: 17
        }
        
        # Define seen/unseen split for zero-shot learning
        self.seen_labels = [0, 1, 2, 3, 4, 5, 6]  # Lying, Sitting, Standing, Walking, Running, Stairs Up/Down
        self.unseen_labels = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # Other activities
        
        # Manual mapping for zero-shot evaluation
        self.manual_mappings = {
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
        
        # Load the dataset
        self.load_data()
    
    def load_data(self):
        """Load and preprocess the PAMAP2 dataset."""
        # 데이터가 이미 로드되었는지 확인
        if not PAMAP2Dataset._is_data_loaded:
            print("Loading PAMAP2 dataset...")
            self._load_raw_data()
        else:
            print("Using cached PAMAP2 dataset...")
            
        # 각 데이터 분할에 따라 적절한 데이터 설정
        self._setup_data_split()
    
    def _load_raw_data(self):
        """Load raw data and cache it for reuse."""
        # Define file paths for protocol and optional data
        data_path_protocol = os.path.join(self.data_path, "Protocol")
        data_path_optional = os.path.join(self.data_path, "Optional")
        
        all_files_protocol = [f for f in os.listdir(data_path_protocol) if f.endswith(".dat")]
        all_files_optional = [f for f in os.listdir(data_path_optional) if f.endswith(".dat")]
        
        # Define column names
        columns = [
            "timestamp", "activityID", "heart_rate",
            "hand_temp", "hand_acc_x", "hand_acc_y", "hand_acc_z",
            "hand_gyro_x", "hand_gyro_y", "hand_gyro_z",
            "hand_mag_x", "hand_mag_y", "hand_mag_z",
            "chest_temp", "chest_acc_x", "chest_acc_y", "chest_acc_z",
            "chest_gyro_x", "chest_gyro_y", "chest_gyro_z",
            "chest_mag_x", "chest_mag_y", "chest_mag_z",
            "ankle_temp", "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
            "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
            "ankle_mag_x", "ankle_mag_y", "ankle_mag_z"
        ]
        
        # Add subject ID column
        columns.append("subjectID")
        
        # Read all data
        all_data = []
        
        # Process protocol data
        for file in all_files_protocol:
            subject_id = int(file.split("subject")[1].split(".dat")[0]) 
            print(f"Processing (Protocol): {file}")
            file_path = os.path.join(data_path_protocol, file)
            with open(file_path, "r") as f:
                reader = csv.reader(f, delimiter=" ")
                for row in reader:
                    row = [x for x in row if x != ""]
                    if len(row) >= len(columns) - 1:
                        data_row = [float(x) for x in row[:len(columns) - 1]]
                        data_row.append(subject_id)
                        all_data.append(data_row)
        
        # Process optional data
        for file in all_files_optional:
            subject_id = int(file.split("subject")[1].split(".dat")[0])
            print(f"Processing (Optional): {file}")
            file_path = os.path.join(data_path_optional, file)
            with open(file_path, "r") as f:
                reader = csv.reader(f, delimiter=" ")
                for row in reader:
                    row = [x for x in row if x != ""]
                    if len(row) >= len(columns) - 1:
                        data_row = [float(x) for x in row[:len(columns) - 1]]
                        data_row.append(subject_id)
                        all_data.append(data_row)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=columns)
        
        # Map original activity IDs to sequential IDs
        df = df[df["activityID"].isin(self.original_label_map.keys())]
        df["activityID"] = df["activityID"].map(self.original_label_map)
        df["activityID"] = pd.to_numeric(df["activityID"]).astype("int32")
        
        # Add activity label column
        df["activityLabel"] = df["activityID"].map(self.label_map)
        
        # Drop unnecessary columns
        df.drop(["timestamp"], axis=1, inplace=True)
        df.drop(["hand_temp", "heart_rate", "chest_temp", "ankle_temp"], axis=1, inplace=True)
        
        # Drop magnetometer data (consistent with notebook)
        df = df[[feature for feature in df.columns if "mag" not in feature]]
        
        # Handle missing values
        for col in df.columns:
            if col not in ['activityID', 'subjectID', 'activityLabel'] and df[col].isnull().any():
                values = df[col].values
                mask = np.isnan(values)
                indices = np.arange(len(values))
                valid_indices = indices[~mask]
                valid_values = values[~mask]
                
                if len(valid_indices) > 0:
                    nearest_indices = np.searchsorted(valid_indices, indices, side='right') - 1
                    nearest_indices = np.maximum(nearest_indices, 0)
                    values[mask] = valid_values[nearest_indices[mask]]
                    
                    if mask[0] and nearest_indices[0] == 0 and mask[nearest_indices[0]]:
                        first_valid = np.argmax(~mask) if np.any(~mask) else -1
                        if first_valid >= 0:
                            values[:first_valid] = values[first_valid]
                
                df[col] = values
        
        # Fill any remaining NaN values - 경고 메시지 수정
        df = df.bfill()  # method='bfill' 대신 bfill() 메서드 직접 호출
        remaining_nulls = df.isnull().sum().sum()
        if remaining_nulls > 0:
            df = df.fillna(df.mean())
        
        # 데이터 캐시 생성
        if self.zero_shot:
            # Split into seen and unseen classes
            df_seen = df[df["activityID"].isin(self.seen_labels)]
            df_unseen = df[df["activityID"].isin(self.unseen_labels)]
            
            # 데이터 저장
            PAMAP2Dataset._cached_data = {
                'df_seen': df_seen,
                'df_unseen': df_unseen
            }
        else:
            PAMAP2Dataset._cached_data = {
                'df': df
            }
            
        PAMAP2Dataset._is_data_loaded = True
    
    def _setup_data_split(self):
        """Set up data according to the requested split."""
        if self.zero_shot:
            self._process_zero_shot_data()
        else:
            self._process_standard_data()
    
    def _process_zero_shot_data(self):
        """
        Process data for zero-shot learning scenario (without SMOTE).
        """
        print("Processing data for zero-shot learning...")
        
        # 캐시된 데이터 사용
        df_seen = PAMAP2Dataset._cached_data['df_seen']
        df_unseen = PAMAP2Dataset._cached_data['df_unseen']
        
        print(f"Seen classes: {len(df_seen)} samples")
        print(f"Unseen classes: {len(df_unseen)} samples")
        
        # Define features and labels
        X_seen = df_seen.drop(["activityID", "subjectID", "activityLabel"], axis=1)
        y_seen = df_seen["activityID"]
        
        X_unseen = df_unseen.drop(["activityID", "subjectID", "activityLabel"], axis=1)
        y_unseen = df_unseen["activityID"]
        
        # Scale the data
        scaler = StandardScaler()
        X_seen_scaled = scaler.fit_transform(X_seen)
        X_unseen_scaled = scaler.transform(X_unseen)
        
        # Save normalization parameters
        self.mean = scaler.mean_
        self.std = scaler.scale_
        
        # Create DataFrames with scaled data
        X_seen_df = pd.DataFrame(X_seen_scaled, columns=X_seen.columns)
        X_unseen_df = pd.DataFrame(X_unseen_scaled, columns=X_unseen.columns)
        
        # Extract accelerometer and gyroscope data for ankle IMU (according to notebook)
        accel_cols = ["ankle_acc_x", "ankle_acc_y", "ankle_acc_z"]
        gyro_cols = ["ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z"]
        
        X_accel_seen = X_seen_df[accel_cols].values
        X_gyro_seen = X_seen_df[gyro_cols].values
        
        X_accel_unseen = X_unseen_df[accel_cols].values
        X_gyro_unseen = X_unseen_df[gyro_cols].values
        
        # Split into windows
        X_accel_seq_seen, y_seq_seen = self.split_windows(X_accel_seen, y_seen.values, 
                                                        overlap=True, clean=self.clean)
        X_gyro_seq_seen, _ = self.split_windows(X_gyro_seen, y_seen.values, 
                                            overlap=True, clean=self.clean)
        
        X_accel_seq_unseen, y_seq_unseen = self.split_windows(X_accel_unseen, y_unseen.values, 
                                                            overlap=True, clean=self.clean)
        X_gyro_seq_unseen, _ = self.split_windows(X_gyro_unseen, y_unseen.values, 
                                                overlap=True, clean=self.clean)
        
        # Split seen data into train and validation
        X_accel_train, X_accel_val, X_gyro_train, X_gyro_val, y_train, y_val = train_test_split(
            X_accel_seq_seen, X_gyro_seq_seen, y_seq_seen, 
            test_size=0.2, random_state=42, stratify=y_seq_seen
        )
        
        # Further split train data to get a test set for seen classes
        X_accel_train, X_accel_test_seen, X_gyro_train, X_gyro_test_seen, y_train, y_test_seen = train_test_split(
            X_accel_train, X_gyro_train, y_train, 
            test_size=0.2, random_state=42, stratify=y_train
        )
        
        # 필요한 속성들 저장
        self.accel_data = None
        self.gyro_data = None
        
        # Store data based on split value
        if self.split == 'train':
            self.accel_data = X_accel_train
            self.gyro_data = X_gyro_train
            self.data = np.concatenate([X_accel_train, X_gyro_train], axis=1)
            self.labels = y_train.astype(np.int32)
        elif self.split == 'val':
            self.accel_data = X_accel_val
            self.gyro_data = X_gyro_val
            self.data = np.concatenate([X_accel_val, X_gyro_val], axis=1)
            self.labels = y_val.astype(np.int32)
        elif self.split == 'test_seen':
            self.accel_data = X_accel_test_seen
            self.gyro_data = X_gyro_test_seen
            self.data = np.concatenate([X_accel_test_seen, X_gyro_test_seen], axis=1)
            self.labels = y_test_seen.astype(np.int32)
        else:  # 'test_unseen'
            self.accel_data = X_accel_seq_unseen
            self.gyro_data = X_gyro_seq_unseen
            self.data = np.concatenate([X_accel_seq_unseen, X_gyro_seq_unseen], axis=1)
            self.labels = y_seq_unseen.astype(np.int32)
        
        # Store all datasets for convenience
        self.train_data = np.concatenate([X_accel_train, X_gyro_train], axis=1)
        self.train_labels = y_train.astype(np.int32)
        
        self.val_data = np.concatenate([X_accel_val, X_gyro_val], axis=1)
        self.val_labels = y_val.astype(np.int32)
        
        self.test_seen_data = np.concatenate([X_accel_test_seen, X_gyro_test_seen], axis=1)
        self.test_seen_labels = y_test_seen.astype(np.int32)
        
        self.test_unseen_data = np.concatenate([X_accel_seq_unseen, X_gyro_seq_unseen], axis=1)
        self.test_unseen_labels = y_seq_unseen.astype(np.int32)
        
        print(f"Train set: {self.train_data.shape}, {self.train_labels.shape}")
        print(f"Validation set: {self.val_data.shape}, {self.val_labels.shape}")
        print(f"Test set (seen): {self.test_seen_data.shape}, {self.test_seen_labels.shape}")
        print(f"Test set (unseen): {self.test_unseen_data.shape}, {self.test_unseen_labels.shape}")
    
    def _process_standard_data(self):
        """
        Process data for standard learning scenario (not zero-shot).
        """
        print("Processing data for standard learning...")
        
        # 캐시된 데이터 사용
        df = PAMAP2Dataset._cached_data['df']
        
        # Split by subject ID
        subject_ids = df["subjectID"].unique()
        
        # Use 70% for training, 10% for validation, 20% for testing
        train_subjects = subject_ids[:int(0.7 * len(subject_ids))]
        val_subjects = subject_ids[int(0.7 * len(subject_ids)):int(0.8 * len(subject_ids))]
        test_subjects = subject_ids[int(0.8 * len(subject_ids)):]
        
        # Create split DataFrames
        df_train = df[df["subjectID"].isin(train_subjects)]
        df_val = df[df["subjectID"].isin(val_subjects)]
        df_test = df[df["subjectID"].isin(test_subjects)]
        
        print(f"Train set: {len(df_train)} samples from subjects {train_subjects}")
        print(f"Validation set: {len(df_val)} samples from subjects {val_subjects}")
        print(f"Test set: {len(df_test)} samples from subjects {test_subjects}")
        
        # Define features and labels
        X_train = df_train.drop(["activityID", "subjectID", "activityLabel"], axis=1)
        y_train = df_train["activityID"]
        
        X_val = df_val.drop(["activityID", "subjectID", "activityLabel"], axis=1)
        y_val = df_val["activityID"]
        
        X_test = df_test.drop(["activityID", "subjectID", "activityLabel"], axis=1)
        y_test = df_test["activityID"]
        
        # Apply SMOTE for class balancing on training data
        if len(df_train) > 0:
            print ("SMOTE")
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        else:
            X_train_smote, y_train_smote = X_train, y_train
        
        # Scale the data
        scaler = StandardScaler()
        X_train_smote_scaled = scaler.fit_transform(X_train_smote)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Save normalization parameters
        self.mean = scaler.mean_
        self.std = scaler.scale_
        
        # Create DataFrames with scaled data
        X_train_smote_df = pd.DataFrame(X_train_smote_scaled, columns=X_train.columns)
        X_val_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)
        X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # Extract accelerometer and gyroscope data for ankle IMU
        accel_cols = ["ankle_acc_x", "ankle_acc_y", "ankle_acc_z"]
        gyro_cols = ["ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z"]
        
        X_accel_train = X_train_smote_df[accel_cols].values
        X_gyro_train = X_train_smote_df[gyro_cols].values
        
        X_accel_val = X_val_df[accel_cols].values
        X_gyro_val = X_val_df[gyro_cols].values
        
        X_accel_test = X_test_df[accel_cols].values
        X_gyro_test = X_test_df[gyro_cols].values
        
        # Split into windows
        X_accel_seq_train, y_seq_train = self.split_windows(X_accel_train, y_train_smote, 
                                                          overlap=True, clean=self.clean)
        X_gyro_seq_train, _ = self.split_windows(X_gyro_train, y_train_smote, 
                                               overlap=True, clean=self.clean)
        
        X_accel_seq_val, y_seq_val = self.split_windows(X_accel_val, y_val.values, 
                                                      overlap=True, clean=self.clean)
        X_gyro_seq_val, _ = self.split_windows(X_gyro_val, y_val.values, 
                                             overlap=True, clean=self.clean)
        
        X_accel_seq_test, y_seq_test = self.split_windows(X_accel_test, y_test.values, 
                                                        overlap=True, clean=self.clean)
        X_gyro_seq_test, _ = self.split_windows(X_gyro_test, y_test.values, 
                                               overlap=True, clean=self.clean)
        
        # 필요한 속성들 초기화
        self.accel_data = None
        self.gyro_data = None
        
        # Store data based on requested split
        if self.split == 'train':
            self.accel_data = X_accel_seq_train
            self.gyro_data = X_gyro_seq_train
            self.data = np.concatenate([X_accel_seq_train, X_gyro_seq_train], axis=1)
            self.labels = y_seq_train.astype(np.int32)
        elif self.split == 'val':
            self.accel_data = X_accel_seq_val
            self.gyro_data = X_gyro_seq_val
            self.data = np.concatenate([X_accel_seq_val, X_gyro_seq_val], axis=1)
            self.labels = y_seq_val.astype(np.int32)
        else:  # 'test'
            self.accel_data = X_accel_seq_test
            self.gyro_data = X_gyro_seq_test
            self.data = np.concatenate([X_accel_seq_test, X_gyro_seq_test], axis=1)
            self.labels = y_seq_test.astype(np.int32)
        
        # Store all datasets for convenience
        self.train_data = np.concatenate([X_accel_seq_train, X_gyro_seq_train], axis=1)
        self.train_labels = y_seq_train.astype(np.int32)
        
        self.val_data = np.concatenate([X_accel_seq_val, X_gyro_seq_val], axis=1)
        self.val_labels = y_seq_val.astype(np.int32)
        
        self.test_data = np.concatenate([X_accel_seq_test, X_gyro_seq_test], axis=1)
        self.test_labels = y_seq_test.astype(np.int32)
        
        print(f"Train set: {self.train_data.shape}, {self.train_labels.shape}")
        print(f"Validation set: {self.val_data.shape}, {self.val_labels.shape}")
        print(f"Test set: {self.test_data.shape}, {self.test_labels.shape}")
    
    def split_windows(self, sequences, labels, overlap=True, clean=True):
        """
        Split sequences into windows.
        
        Args:
            sequences: Data sequences
            labels: Labels for each sequence step
            overlap: Whether to use overlapping windows
            clean: Whether to filter windows with mixed activities
        
        Returns:
            X_windows: Windowed data
            y_windows: Labels for each window
        """
        window_size = self.window_width
        if overlap:
            stride = self.stride
        else:
            stride = window_size
        
        n_samples, n_features = sequences.shape
        n_windows = (n_samples - window_size) // stride + 1
        
        X_windows = np.zeros((n_windows, window_size, n_features))
        y_windows = np.zeros(n_windows)
        
        window_idx = 0
        for i in range(0, n_samples - window_size + 1, stride):
            X_windows[window_idx] = sequences[i:i + window_size]
            
            # Use most frequent label in window if clean=True, otherwise use last label
            if clean:
                window_labels = labels[i:i + window_size]
                unique_labels, counts = np.unique(window_labels, return_counts=True)
                
                # Skip windows with mixed activities
                if len(unique_labels) > 1:
                    most_common_label = unique_labels[np.argmax(counts)]
                    most_common_count = np.max(counts)
                    
                    # If the most common label doesn't dominate (e.g., >80%), skip the window
                    if most_common_count < 0.8 * window_size:
                        continue
                    
                    y_windows[window_idx] = most_common_label
                else:
                    y_windows[window_idx] = unique_labels[0]
            else:
                # Use the label of the last sample in the window
                y_windows[window_idx] = labels[i + window_size - 1]
            
            window_idx += 1
        
        # Trim any unused windows
        if window_idx < n_windows:
            X_windows = X_windows[:window_idx]
            y_windows = y_windows[:window_idx]
        
        return X_windows, y_windows
    
    def get_tf_dataset(self, batch_size=64, shuffle=True):
        """
        Convert the data to a TensorFlow dataset.
        
        Args:
            batch_size (int): Batch size for the dataset
            shuffle (bool): Whether to shuffle the dataset
            
        Returns:
            tf.data.Dataset: TensorFlow dataset
        """
        import tensorflow as tf
        
        # Create a dictionary with accel and gyro data
        features = {
            'accel': self.accel_data,
            'gyro': self.gyro_data
        }
        
        # Create a TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, self.labels))
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.labels))
        
        # Batch the dataset
        dataset = dataset.batch(batch_size)
        
        return dataset