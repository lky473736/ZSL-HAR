#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logging utility for the Zero-Shot HAR system.
Provides functions for logging messages, results, and metrics.
"""

import os
import logging
import datetime
import numpy as np
import pandas as pd
import json
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

class Logger:
    """Logger for training, evaluation, and testing."""
    
    def __init__(self, log_dir, name="zeroshot_har", verbose=True):
        """
        Initialize the logger.
        
        Args:
            log_dir (str): Directory for log files
            name (str): Logger name
            verbose (bool): Whether to print logs to console
        """
        self.log_dir = log_dir
        self.name = name
        self.verbose = verbose
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear existing handlers
        
        # Create file handler
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", 
                                    datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        self.logger.addHandler(file_handler)
        
        # Add console handler if verbose is True
        if verbose:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
        self.info(f"Logger initialized. Log file: {log_file}")
    
    def info(self, message):
        """Log an info message."""
        if self.verbose:
            print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} {message}")
        self.logger.info(message)
    
    def warning(self, message):
        """Log a warning message."""
        if self.verbose:
            print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} {message}")
        self.logger.warning(message)
    
    def error(self, message):
        """Log an error message."""
        if self.verbose:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {message}")
        self.logger.error(message)
    
    def log_config(self, config):
        """
        Log configuration parameters.
        
        Args:
            config (dict or object): Configuration parameters
        """
        self.info("Configuration parameters:")
        
        if isinstance(config, dict):
            # If config is a dictionary
            for key, value in config.items():
                self.info(f"  {key}: {value}")
                
            # Save configuration to file
            config_file = os.path.join(self.log_dir, "config.json")
            with open(config_file, "w") as f:
                json.dump(config, f, indent=4)
                
        else:
            # If config is an object with attributes
            for key, value in vars(config).items():
                if not key.startswith("_"):
                    self.info(f"  {key}: {value}")
            
            # Save configuration to file
            config_file = os.path.join(self.log_dir, "config.txt")
            with open(config_file, "w") as f:
                for key, value in vars(config).items():
                    if not key.startswith("_"):
                        f.write(f"{key}: {value}\n")
        
        self.info(f"Configuration saved to {config_file}")
    
    def log_metrics(self, metrics, prefix=""):
        """
        Log evaluation metrics.
        
        Args:
            metrics (dict): Dictionary of metrics
            prefix (str): Prefix for metric names
        """
        self.info(f"{prefix} Metrics:")
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items() 
                                if isinstance(v, (int, float))})
        
        metrics_file = os.path.join(self.log_dir, f"{prefix.lower()}_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        
        self.info(f"Metrics saved to {metrics_file}")
    
    def log_confusion_matrix(self, confusion_matrix, class_names, title="confusion_matrix"):
        """
        Log confusion matrix.
        
        Args:
            confusion_matrix (numpy.ndarray): Confusion matrix
            class_names (list): List of class names
            title (str): Title for the confusion matrix file
        """
        self.info(f"Saving confusion matrix to {self.log_dir}/{title}.txt")
        
        # Save confusion matrix to text file
        cm_file = os.path.join(self.log_dir, f"{title}.txt")
        with open(cm_file, "w") as f:
            f.write("Confusion Matrix:\n\n")
            
            # Write class names as header
            f.write("True\\Pred")
            for name in class_names:
                f.write(f",{name}")
            f.write("\n")
            
            # Write confusion matrix values
            for i, row in enumerate(confusion_matrix):
                f.write(f"{class_names[i]}")
                for value in row:
                    f.write(f",{value}")
                f.write("\n")
        
        # Save as CSV for easier analysis
        cm_df = pd.DataFrame(confusion_matrix, 
                           index=class_names, 
                           columns=class_names)
        cm_df.to_csv(os.path.join(self.log_dir, f"{title}.csv"))
    
    def log_training_progress(self, epoch, epochs, metrics, validation=False):
        """
        Log training progress.
        
        Args:
            epoch (int): Current epoch
            epochs (int): Total number of epochs
            metrics (dict): Dictionary of metrics
            validation (bool): Whether these are validation metrics
        """
        prefix = "Validation" if validation else "Training"
        
        # Format metrics string
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        self.info(f"Epoch {epoch+1}/{epochs} - {prefix}: {metrics_str}")
    
    def log_class_performance(self, per_class_metrics, title="per_class_metrics"):
        """
        Log per-class performance metrics.
        
        Args:
            per_class_metrics (pandas.DataFrame): DataFrame of per-class metrics
            title (str): Title for the metrics file
        """
        self.info(f"Per-class performance metrics:")
        
        for _, row in per_class_metrics.iterrows():
            class_id = row["class_id"]
            class_name = row["class_name"]
            accuracy = row["accuracy"]
            f1 = row["f1"]
            
            self.info(f"  Class {class_id} ({class_name}): Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        # Save to CSV
        per_class_metrics.to_csv(os.path.join(self.log_dir, f"{title}.csv"), index=False)
        
        self.info(f"Per-class metrics saved to {self.log_dir}/{title}.csv")
    
    def log_mapping_results(self, mapping_type, metrics):
        """
        Log zero-shot mapping results.
        
        Args:
            mapping_type (str): Type of mapping (e.g., "Manual", "Top-1")
            metrics (dict): Dictionary of metrics
        """
        self.info(f"Zero-shot mapping ({mapping_type}) results:")
        
        for key, value in metrics.items():
            self.info(f"  {key}: {value:.4f}")
        
        # Save to CSV
        mapping_file = os.path.join(self.log_dir, f"mapping_{mapping_type.lower()}.csv")
        pd.DataFrame([metrics]).to_csv(mapping_file, index=False)
        
        self.info(f"Mapping results saved to {mapping_file}")

def setup_logger(output_dir, name="zeroshot_har"):
    """
    Set up logger for the Zero-Shot HAR system.
    
    Args:
        output_dir (str): Output directory
        name (str): Logger name
        
    Returns:
        Logger: Logger instance
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logger
    return Logger(output_dir, name)

if __name__ == "__main__":
    # Test logger
    logger = setup_logger("test_logs")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test logging metrics
    metrics = {"accuracy": 0.85, "f1": 0.83, "precision": 0.82, "recall": 0.84}
    logger.log_metrics(metrics, prefix="Test")