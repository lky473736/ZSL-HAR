#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization utility for the Zero-Shot HAR system.
Provides functions for creating plots and visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap

def plot_activity_distribution(labels, label_map=None, title="Activity Distribution", 
                             figsize=(12, 6), save_path=None):
    """
    Plot activity distribution.
    
    Args:
        labels (numpy.ndarray): Activity labels
        label_map (dict): Mapping from label to class name
        title (str): Plot title
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    plt.figure(figsize=figsize)
    
    # Get label counts
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Get class names
    if label_map:
        class_names = [label_map.get(label, f"Class {label}") for label in unique_labels]
    else:
        class_names = [f"Class {label}" for label in unique_labels]
    
    # Create color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    # Plot bar chart
    bars = plt.bar(class_names, counts, color=colors)
    
    # Add counts on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f"{int(height)}", ha="center", va="bottom")
    
    plt.title(title, fontsize=16)
    plt.xlabel("Activity", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return plt.gcf()

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", 
                        figsize=(12, 10), normalize=False, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        class_names (list): List of class names
        title (str): Plot title
        figsize (tuple): Figure size
        normalize (bool): Whether to normalize confusion matrix
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    plt.figure(figsize=figsize)
    
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Replace NaN with zero
        fmt = ".2f"
    else:
        fmt = "d"
    
    cm = cm.astype(int)

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
              xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title, fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return plt.gcf()

def plot_training_history(history, metrics=["loss", "accuracy"], 
                        figsize=(15, 5), save_path=None):
    """
    Plot training history.
    
    Args:
        history (dict): Training history dictionary
        metrics (list): List of metrics to plot
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    n_metrics = len(metrics)
    plt.figure(figsize=(figsize[0], figsize[1] * n_metrics))
    
    for i, metric in enumerate(metrics):
        plt.subplot(n_metrics, 1, i+1)
        
        if metric in history:
            plt.plot(history[metric], label=f"Training {metric}")
        
        val_metric = f"val_{metric}"
        if val_metric in history:
            plt.plot(history[val_metric], label=f"Validation {metric}")
        
        plt.title(f"{metric.capitalize()} over Epochs", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return plt.gcf()

def plot_embeddings_tsne(embeddings, labels, label_map=None, perplexity=30, 
                       title="t-SNE Visualization", figsize=(12, 10), save_path=None):
    """
    Plot embeddings using t-SNE.
    
    Args:
        embeddings (numpy.ndarray): Embeddings
        labels (numpy.ndarray): Labels
        label_map (dict): Mapping from label to class name
        perplexity (int): Perplexity parameter for t-SNE
        title (str): Plot title
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    plt.figure(figsize=figsize)
    
    # Get unique labels
    unique_labels = np.unique(labels)
    
    # Get class names
    if label_map:
        class_names = {label: label_map.get(label, f"Class {label}") for label in unique_labels}
    else:
        class_names = {label: f"Class {label}" for label in unique_labels}
    
    # Create color palette
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot each class
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   color=colors[i], label=class_names[label], alpha=0.7)
    
    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return plt.gcf()

def plot_embeddings_umap(embeddings, labels, label_map=None, n_neighbors=15, min_dist=0.1,
                      title="UMAP Visualization", figsize=(12, 10), save_path=None):
    """
    Plot embeddings using UMAP.
    
    Args:
        embeddings (numpy.ndarray): Embeddings
        labels (numpy.ndarray): Labels
        label_map (dict): Mapping from label to class name
        n_neighbors (int): Number of neighbors parameter for UMAP
        min_dist (float): Minimum distance parameter for UMAP
        title (str): Plot title
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    plt.figure(figsize=figsize)
    
    # Get unique labels
    unique_labels = np.unique(labels)
    
    # Get class names
    if label_map:
        class_names = {label: label_map.get(label, f"Class {label}") for label in unique_labels}
    else:
        class_names = {label: f"Class {label}" for label in unique_labels}
    
    # Create color palette
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Run UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Plot each class
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   color=colors[i], label=class_names[label], alpha=0.7)
    
    plt.title(title, fontsize=16)
    plt.xlabel("UMAP Component 1", fontsize=12)
    plt.ylabel("UMAP Component 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return plt.gcf()

def plot_similarity_matrix(similarity_matrix, row_labels, col_labels, 
                         row_label_map=None, col_label_map=None, 
                         title="Similarity Matrix", figsize=(12, 10), save_path=None):
    """
    Plot similarity matrix.
    
    Args:
        similarity_matrix (numpy.ndarray): Similarity matrix
        row_labels (list): Row labels
        col_labels (list): Column labels
        row_label_map (dict): Mapping from row label to class name
        col_label_map (dict): Mapping from column label to class name
        title (str): Plot title
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    plt.figure(figsize=figsize)
    
    # Get row and column names
    if row_label_map:
        row_names = [row_label_map.get(label, f"Class {label}") for label in row_labels]
    else:
        row_names = [f"Class {label}" for label in row_labels]
        
    if col_label_map:
        col_names = [col_label_map.get(label, f"Class {label}") for label in col_labels]
    else:
        col_names = [f"Class {label}" for label in col_labels]
    
    # Plot heatmap
    sns.heatmap(similarity_matrix, annot=True, fmt=".3f", cmap="YlOrRd",
              xticklabels=col_names, yticklabels=row_names)
    
    plt.title(title, fontsize=16)
    plt.xlabel("Column Labels", fontsize=12)
    plt.ylabel("Row Labels", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return plt.gcf()

def plot_adjacency_matrix(adjacency_matrix, row_labels, col_labels, 
                        mapping_dict=None, row_label_map=None, col_label_map=None,
                        title="Adjacency Matrix", figsize=(12, 10), save_path=None):
    """
    Plot adjacency matrix for zero-shot mapping.
    
    Args:
        adjacency_matrix (numpy.ndarray): Adjacency matrix
        row_labels (list): Row labels (unseen classes)
        col_labels (list): Column labels (seen classes)
        mapping_dict (dict): Mapping from unseen to seen classes
        row_label_map (dict): Mapping from row label to class name
        col_label_map (dict): Mapping from column label to class name
        title (str): Plot title
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    plt.figure(figsize=figsize)
    
    # Get row and column names
    if row_label_map:
        row_names = [row_label_map.get(label, f"Class {label}") for label in row_labels]
    else:
        row_names = [f"Class {label}" for label in row_labels]
        
    if col_label_map:
        col_names = [col_label_map.get(label, f"Class {label}") for label in col_labels]
    else:
        col_names = [f"Class {label}" for label in col_labels]
    
    # Plot heatmap
    adjacency_matrix = adjacency_matrix.astype(int)
    ax = sns.heatmap(adjacency_matrix, annot=True, fmt="d", cmap="Blues",
                   xticklabels=col_names, yticklabels=row_names)
    
    # Highlight mapping connections
    if mapping_dict:
        for i, row_label in enumerate(row_labels):
            if row_label in mapping_dict:
                for col_label in mapping_dict[row_label]:
                    if col_label in col_labels:
                        j = col_labels.index(col_label)
                        # Draw red rectangle around the cell
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2))
    
    plt.title(title, fontsize=16)
    plt.xlabel("Seen Classes", fontsize=12)
    plt.ylabel("Unseen Classes", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return plt.gcf()

def plot_metrics_comparison(s_metrics, u_metrics, h_metrics, 
                          title="S-U-H Metrics Comparison", 
                          figsize=(12, 6), save_path=None):
    """
    Plot comparison of seen, unseen, and harmonic metrics.
    
    Args:
        s_metrics (dict): Metrics for seen classes
        u_metrics (dict): Metrics for unseen classes
        h_metrics (dict): Harmonic mean metrics
        title (str): Plot title
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    plt.figure(figsize=figsize)
    
    # Get metric names
    metrics = list(s_metrics.keys())
    
    # Plot grouped bars
    x = np.arange(len(metrics))
    width = 0.25
    
    plt.bar(x - width, [s_metrics[m] for m in metrics], width, label="Seen (S)", color="green")
    plt.bar(x, [u_metrics[m] for m in metrics], width, label="Unseen (U)", color="blue")
    plt.bar(x + width, [h_metrics[m] for m in metrics], width, label="Harmonic (H)", color="red")
    
    # Add value labels
    for i, m in enumerate(metrics):
        plt.text(i - width, s_metrics[m] + 0.02, f"{s_metrics[m]:.3f}", ha="center", va="bottom", fontsize=8)
        plt.text(i, u_metrics[m] + 0.02, f"{u_metrics[m]:.3f}", ha="center", va="bottom", fontsize=8)
        plt.text(i + width, h_metrics[m] + 0.02, f"{h_metrics[m]:.3f}", ha="center", va="bottom", fontsize=8)
    
    plt.title(title, fontsize=16)
    plt.xlabel("Metric", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.xticks(x, [m.capitalize() for m in metrics])
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return plt.gcf()

def plot_sensor_data(data, window_idx=0, title="Sensor Data", 
                   figsize=(12, 4), save_path=None):
    """
    Plot sensor data for a specific window.
    
    Args:
        data (numpy.ndarray): Sensor data of shape (n_windows, window_width, n_channels)
        window_idx (int): Window index to plot
        title (str): Plot title
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    plt.figure(figsize=figsize)
    
    window_data = data[window_idx]
    window_width = window_data.shape[0]
    n_channels = window_data.shape[1]
    
    # Create time axis
    time = np.arange(window_width)
    
    # Plot each channel
    for i in range(n_channels):
        plt.plot(time, window_data[:, i], label=f"Channel {i+1}")
    
    plt.title(title, fontsize=16)
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return plt.gcf()

if __name__ == "__main__":
    # Test visualization functions
    labels = np.random.randint(0, 5, 100)
    plot_activity_distribution(labels, title="Random Activity Distribution")
    plt.show()