#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation metrics utility for the Zero-Shot HAR system.
Provides functions for calculating metrics and evaluating models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)

def calculate_metrics(y_true, y_pred, average="weighted"):
    """
    Calculate classification metrics.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        average (str): Averaging method for metrics
        
    Returns:
        dict: Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def calculate_per_class_metrics(y_true, y_pred, label_map=None):
    """
    Calculate per-class metrics.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        label_map (dict): Mapping from label to class name
        
    Returns:
        pandas.DataFrame: DataFrame of per-class metrics
    """
    # Get unique classes
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # Create metrics list
    metrics_list = []
    
    for cls in unique_classes:
        # Calculate binary metrics
        true_positives = np.sum((y_true == cls) & (y_pred == cls))
        false_positives = np.sum((y_true != cls) & (y_pred == cls))
        false_negatives = np.sum((y_true == cls) & (y_pred != cls))
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Get class name
        class_name = label_map[cls] if label_map and cls in label_map else f"Class {cls}"
        
        # Total samples for this class
        total_samples = np.sum(y_true == cls)
        
        # Accuracy for this class
        accuracy = true_positives / total_samples if total_samples > 0 else 0
        
        # Add to metrics list
        metrics_list.append({
            "class_id": int(cls),
            "class_name": class_name,
            "samples": int(total_samples),
            "tp": int(true_positives),
            "fp": int(false_positives),
            "fn": int(false_negatives),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        })
    
    # Create DataFrame
    return pd.DataFrame(metrics_list)

def get_confusion_matrix(y_true, y_pred, normalize=False):
    """
    Calculate confusion matrix.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        normalize (bool): Whether to normalize confusion matrix
        
    Returns:
        numpy.ndarray: Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Replace NaN with zero
    
    return cm

def evaluate_zero_shot_mapping(y_true, y_pred, mapping_dict):
    """
    Evaluate zero-shot mapping strategy.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        mapping_dict (dict): Mapping from unseen to seen classes
        
    Returns:
        dict: Dictionary of metrics
    """
    total_samples = len(y_true)
    correct_count = 0
    
    # Create binary arrays for metrics calculation
    y_true_binary = np.ones(total_samples)  # All should be mapped correctly
    y_pred_binary = np.zeros(total_samples)  # Start with all incorrect
    
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if pred_label in mapping_dict.get(true_label, []):
            correct_count += 1
            y_pred_binary[i] = 1  # Mark as correct if in mapping
    
    # Calculate metrics
    accuracy = correct_count / total_samples
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def calculate_topk_mappings(similarity_matrix, seen_labels, unseen_labels, k=1):
    """
    Calculate top-k mappings based on similarity matrix.
    
    Args:
        similarity_matrix (numpy.ndarray): Similarity matrix
        seen_labels (list): List of seen class labels
        unseen_labels (list): List of unseen class labels
        k (int): Number of top matches to consider
        
    Returns:
        dict: Mapping from unseen to seen classes
    """
    topk_mappings = {}
    
    for i, unseen_label in enumerate(unseen_labels):
        similarities = similarity_matrix[i]
        top_indices = np.argsort(similarities)[-k:][::-1]  # Get indices of top k similarities
        top_seen_labels = [seen_labels[idx] for idx in top_indices]
        topk_mappings[unseen_label] = top_seen_labels
    
    return topk_mappings

def calculate_harmonic_mean(seen_metrics, unseen_metrics):
    """
    Calculate harmonic mean between seen and unseen metrics.
    
    Args:
        seen_metrics (dict): Metrics for seen classes
        unseen_metrics (dict): Metrics for unseen classes
        
    Returns:
        dict: Harmonic mean metrics
    """
    harmonic_metrics = {}
    
    for key in seen_metrics:
        if key in unseen_metrics:
            s_value = seen_metrics[key]
            u_value = unseen_metrics[key]
            
            if s_value > 0 and u_value > 0:
                harmonic_metrics[key] = 2 * s_value * u_value / (s_value + u_value)
            else:
                harmonic_metrics[key] = 0
    
    return harmonic_metrics

def evaluate_seen_unseen(y_seen_true, y_seen_pred, y_unseen_true, y_unseen_pred, 
                       mapping_dict, average="weighted"):
    """
    Evaluate model on both seen and unseen classes.
    
    Args:
        y_seen_true (numpy.ndarray): True labels for seen classes
        y_seen_pred (numpy.ndarray): Predicted labels for seen classes
        y_unseen_true (numpy.ndarray): True labels for unseen classes
        y_unseen_pred (numpy.ndarray): Predicted labels for unseen classes
        mapping_dict (dict): Mapping from unseen to seen classes
        average (str): Averaging method for metrics
        
    Returns:
        tuple: (seen_metrics, unseen_metrics, harmonic_metrics)
    """
    # Calculate metrics for seen classes
    seen_metrics = calculate_metrics(y_seen_true, y_seen_pred, average=average)
    
    # Calculate metrics for unseen classes using mapping
    unseen_metrics = evaluate_zero_shot_mapping(y_unseen_true, y_unseen_pred, mapping_dict)
    
    # Calculate harmonic mean
    harmonic_metrics = calculate_harmonic_mean(seen_metrics, unseen_metrics)
    
    return seen_metrics, unseen_metrics, harmonic_metrics

if __name__ == "__main__":
    # Test metrics calculation
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 2, 0, 0, 2, 3])
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    print("Overall metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Calculate per-class metrics
    per_class_metrics = calculate_per_class_metrics(y_true, y_pred)
    print("\nPer-class metrics:")
    print(per_class_metrics)
    
    # Calculate confusion matrix
    cm = get_confusion_matrix(y_true, y_pred)
    print("\nConfusion matrix:")
    print(cm)
    
    # Test zero-shot mapping
    mapping_dict = {0: [0, 1], 1: [1], 2: [2, 3], 3: [3]}
    mapping_metrics = evaluate_zero_shot_mapping(y_true, y_pred, mapping_dict)
    print("\nZero-shot mapping metrics:")
    for key, value in mapping_metrics.items():
        print(f"  {key}: {value:.4f}")