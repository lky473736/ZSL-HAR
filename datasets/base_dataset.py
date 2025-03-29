#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base class for all HAR datasets.
Provides common functionality for loading, preprocessing, and windowing data.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from collections import Counter
from scipy.signal import butter, lfilter

class BaseDataset(ABC):
    """Base class for HAR datasets with common preprocessing methods."""
    
    def __init__(self, data_path, window_width=128, stride=64, sampling_rate=50):
        """
        Initialize the dataset.
        
        Args:
            data_path (str): Path to the dataset
            window_width (int): Window width for segmentation
            stride (int): Stride for sliding window
            sampling_rate (int): Sampling rate in Hz
        """
        self.data_path = data_path
        self.window_width = window_width
        self.stride = stride
        self.sampling_rate = sampling_rate
        
        # These will be set by subclasses
        self.train_data = None
        self.train_label = None
        self.val_data = None
        self.val_label = None
        self.test_data = None
        self.test_label = None
        
        # Normalization parameters
        self.mean = None
        self.std = None
        
        # Label mapping
        self.label_map = {}
    
    @abstractmethod
    def load_data(self):
        """Load the dataset. To be implemented by subclasses."""
        pass
    
    @staticmethod
    def butterworth_filter(data, fs, lowcut=20, order=3):
        """
        Apply Butterworth low-pass filter to data.
        
        Args:
            data (numpy.ndarray): Data to filter
            fs (int): Sampling frequency
            lowcut (int): Cutoff frequency
            order (int): Filter order
            
        Returns:
            numpy.ndarray: Filtered data
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        b, a = butter(order, low, btype='low')
        
        if len(data.shape) > 1:
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[1]):
                filtered_data[:, i] = lfilter(b, a, data[:, i])
            return filtered_data
        else:
            return lfilter(b, a, data)
    
    def split_windows(self, raw_data, raw_label, overlap=True, clean=True):
        """
        Split data into windows.
        
        Args:
            raw_data (numpy.ndarray): Raw data
            raw_label (numpy.ndarray): Raw labels
            overlap (bool): Whether to use overlapping windows
            clean (bool): Whether to filter out windows with mixed activities
            
        Returns:
            tuple: (windowed_data, windowed_labels)
        """
        idx = 0
        endidx = len(raw_data)
        data = []
        label = []
        
        while idx < endidx - self.window_width:
            data_segment = raw_data[idx:idx+self.window_width].T
            
            if clean and len(np.unique(raw_label[idx:idx + self.window_width])) > 1:
                # Skip windows with multiple activities if clean=True
                pass
            else:
                data.append(data_segment)
                label.append(raw_label[idx+self.window_width-1])  # Use the last label in the window
                
            if overlap:
                idx += self.stride
            else:
                idx += self.window_width
                
        if len(data) == 0:
            return None, None
            
        return np.stack(data), np.asarray(label)
    
    def normalize(self, train=True):
        """
        Normalize data using train set mean and std.
        
        Args:
            train (bool): Whether to fit scaler on train data
        """
        if train or (self.mean is None or self.std is None):
            # Reshape train data for normalization
            samples = self.train_data.transpose(1, 0, 2).reshape(self.train_data.shape[1], -1)
            self.mean = np.mean(samples, axis=1)
            self.std = np.std(samples, axis=1)
        
        # Apply normalization
        self.train_data = (self.train_data - self.mean.reshape(1, -1, 1)) / self.std.reshape(1, -1, 1)
        
        if self.val_data is not None:
            self.val_data = (self.val_data - self.mean.reshape(1, -1, 1)) / self.std.reshape(1, -1, 1)
            
        if self.test_data is not None:
            self.test_data = (self.test_data - self.mean.reshape(1, -1, 1)) / self.std.reshape(1, -1, 1)
    
    def dataset_verbose(self):
        """Print dataset information."""
        print(f"\n--- Dataset: {self.__class__.__name__} ---")
        
        if self.train_data is not None:
            print(f"# train: {len(self.train_data)}")
            train_counts = dict(Counter(self.train_label))
            print(f"Train class distribution: {sorted(train_counts.items())}")
            
        if self.val_data is not None:
            print(f"# val: {len(self.val_data)}")
            val_counts = dict(Counter(self.val_label))
            print(f"Validation class distribution: {sorted(val_counts.items())}")
            
        if self.test_data is not None:
            print(f"# test: {len(self.test_data)}")
            test_counts = dict(Counter(self.test_label))
            print(f"Test class distribution: {sorted(test_counts.items())}")
            
        print(f"Window width: {self.window_width}, Stride: {self.stride}")
        print(f"Sampling rate: {self.sampling_rate} Hz")
    
    def get_tf_dataset(self, split='train', batch_size=64, shuffle=True):
        """
        Create TensorFlow dataset for a specific split.
        
        Args:
            split (str): Split to create dataset for ('train', 'val', or 'test')
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle the dataset
            
        Returns:
            tf.data.Dataset: TensorFlow dataset
        """
        if split == 'train':
            data, labels = self.train_data, self.train_label
        elif split == 'val':
            data, labels = self.val_data, self.val_label
        elif split == 'test':
            data, labels = self.test_data, self.test_label
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Get feature dimensions
        n_features = data.shape[1]
        n_channels = n_features // 2  # Assuming half accelerometer, half gyroscope
        
        # Split data into accelerometer and gyroscope components
        accel_data = data[:, :n_channels, :]
        gyro_data = data[:, n_channels:, :]
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            ({"accel": accel_data, "gyro": gyro_data}, labels)
        )
        
        # Apply shuffling if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(data))
            
        # Apply batching and prefetching
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        return dataset