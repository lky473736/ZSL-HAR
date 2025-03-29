#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PAMAP2 dataset handler for Zero-Shot HAR.
Loads and processes the PAMAP2 dataset with zero-shot learning capabilities.
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
    
    def __init__(self, data_path, window_width=128, stride=64, 
                 clean=True, include_null=True, zero_shot=True, split='train'):
        """
        Initialize the PAMAP2 dataset.
        
        Args:
            data_path (str): Path to the PAMAP2 dataset
            window_width (int): Window width for segmentation
            stride (int): Stride for sliding window
            clean (bool): Whether to filter out windows with mixed activities
            include_null (bool): Whether to include null class (activity ID 0)
            zero_shot (bool): Whether to use zero-shot learning setup
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
        
        # Original -> sequential ID mapping
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
        print("Loading PAMAP2 dataset...")
        
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
        
        # Drop magnetometer data
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
        
        # Fill any remaining NaN values
        df = df.fillna(method='bfill')
        df = df.fillna(df.mean())
        
        # Process data according to zero-shot setting
        if self.zero_shot:
            self._process_zero_shot_data(df)
        else:
            self._process_standard_data(df)
    
    def _process_zero_shot_data(self, df):
        """
        Process data for zero-shot learning scenario.
        
        Args:
            df (pandas.DataFrame): Preprocessed DataFrame
        """
        print("Processing data for zero-shot learning...")
        
        # Split into seen and unseen classes
        df_seen = df[df["activityID"].isin(self.seen_labels)]
        df_unseen = df[df["activityID"].isin(self.unseen_labels)]
        
        print(f"Seen classes: {len(df_seen)} samples")
        print(f"Unseen classes: {len(df_unseen)} samples")
        
        # Define features and labels
        X_seen = df_seen.drop(["activityID", "subjectID", "activityLabel"], axis=1)
        y_seen = df_seen["activityID"]
        
        X_unseen = df_unseen.drop(["activityID", "subjectID", "activityLabel"], axis=1)
        y_unseen = df_unseen["activityID"]
        
        # Apply SMOTE for class balancing on seen classes
        smote = SMOTE(random_state=42)
        X_seen_smote, y_seen_smote = smote.fit_resample(X_seen, y_seen)
        
        # Scale the data
        scaler = StandardScaler()
        X_seen_smote_scaled = scaler.fit_transform(X_seen_smote)
        X_unseen_scaled = scaler.transform(X_unseen)
        
        # Save normalization parameters
        self.mean = scaler.mean_
        self.std = scaler.scale_
        
        # Create DataFrames with scaled data
        X_seen_smote_df = pd.DataFrame(X_seen_smote_scaled, columns=X_seen.columns)
        X_unseen_df = pd.DataFrame(X_unseen_scaled, columns=X_unseen.columns)
        
        # Extract accelerometer and gyroscope data for ankle IMU (according to your original code)
        accel_cols = ["ankle_acc_x", "ankle_acc_y", "ankle_acc_z"]
        gyro_cols = ["ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z"]
        
        X_accel_seen_smote = X_seen_smote_df[accel_cols].values
        X_gyro_seen_smote = X_seen_smote_df[gyro_cols].values
        
        X_accel_unseen = X_unseen_df[accel_cols].values
        X_gyro_unseen = X_unseen_df[gyro_cols].values
        
        # Split into windows
        X_accel_seq_seen, y_seq_seen = self.split_windows(X_accel_seen_smote, y_seen_smote, 
                                                        overlap=True, clean=self.clean)
        X_gyro_seq_seen, _ = self.split_windows(X_gyro_seen_smote, y_seen_smote, 
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
        
        # 데이터 저장 - split 값에 따라 다른 처리
        if self.split == 'train':
            # 훈련 데이터만 설정
            self.data = np.concatenate([X_accel_train, X_gyro_train], axis=1)
            self.labels = y_train.astype(np.int32)
        elif self.split == 'val':
            # 검증 데이터만 설정
            self.data = np.concatenate([X_accel_val, X_gyro_val], axis=1)
            self.labels = y_val.astype(np.int32)
        else:  # 'test'
            # 테스트 데이터만 설정
            self.data = np.concatenate([X_accel_seq_unseen, X_gyro_seq_unseen], axis=1)
            self.labels = y_seq_unseen.astype(np.int32)
        
        print(f"Train set: {self.train_data.shape}, {self.train_label.shape}")
        print(f"Validation set: {self.val_data.shape}, {self.val_label.shape}")
        print(f"Test set (unseen): {self.test_data.shape}, {self.test_label.shape}")
    
    def _process_standard_data(self, df):
        """
        Process data for standard learning scenario (not zero-shot).
        
        Args:
            df (pandas.DataFrame): Preprocessed DataFrame
        """
        print("Processing data for standard learning...")
        
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
        
        # Store data
        self.train_data = np.concatenate([X_accel_seq_train, X_gyro_seq_train], axis=1)
        self.train_label = y_seq_train.astype(np.int32)
        
        self.val_data = np.concatenate([X_accel_seq_val, X_gyro_seq_val], axis=1)
        self.val_label = y_seq_val.astype(np.int32)
        
        self.test_data = np.concatenate([X_accel_seq_test, X_gyro_seq_test], axis=1)
        self.test_label = y_seq_test.astype(np.int32)
        
        print(f"Train set: {self.train_data.shape}, {self.train_label.shape}")
        print(f"Validation set: {self.val_data.shape}, {self.val_label.shape}")
        print(f"Test set: {self.test_data.shape}, {self.test_label.shape}")

if __name__ == "__main__":
    # Test the dataset loading
    data_path = os.path.join("data", "PAMAP2_Dataset")
    dataset = PAMAP2Dataset(data_path, zero_shot=True)
    dataset.dataset_verbose()