#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Architecture: Temporal Convolutional Network (TCN) + BiGRU + Transformer for HAR
Implements the zero-shot learning architecture for activity recognition.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

# Custom L2Normalization layer
class L2NormalizationLayer(layers.Layer):
    """Layer that normalizes inputs to unit L2 norm."""
    
    def __init__(self, axis=1, **kwargs):
        super(L2NormalizationLayer, self).__init__(**kwargs)
        self.axis = axis
        
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=self.axis)

# Custom Cosine Similarity Layer (following paper equation 2 and 14)
class SimilarityLayer(layers.Layer):
    """
    Layer that calculates cosine similarity matrix as per paper equation (14):
    S_{i,j} = (f^(a)_i · f^(g)_j) / (||f^(a)_i|| · ||f^(g)_j||)
    
    Since inputs are already L2-normalized, this simplifies to:
    S_{i,j} = f^(a)_i · f^(g)_j (batch-wise dot product)
    """
    
    def call(self, inputs):
        f_accel, f_gyro = inputs
        
        # Since inputs are already L2-normalized in the encoder,
        # cosine similarity = dot product
        # Paper equation (2): sim(f^(a), f^(g)) = (f^(a) · f^(g)) / (||f^(a)||_2 ||f^(g)||_2)
        similarity_matrix = tf.matmul(f_accel, f_gyro, transpose_b=True)
        
        return similarity_matrix

# TCN Block
def TCN_Block(x, filters, kernel_size=3, dilation_rate=1, prefix=""):
    """
    Temporal Convolutional Network Block.
    
    Args:
        x (tf.Tensor): Input tensor
        filters (int): Number of filters
        kernel_size (int): Kernel size
        dilation_rate (int): Dilation rate
        prefix (str): Prefix for layer names
        
    Returns:
        tf.Tensor: Output tensor
    """
    # Residual connection
    residual = x
    
    # If input and output dimensions don't match, use 1x1 conv for residual
    if int(x.shape[-1]) != filters:
        residual = layers.Conv1D(filters, 1, padding="same", name=f"{prefix}_residual_conv")(x)
    
    # First dilated causal conv
    x = layers.Conv1D(filters, kernel_size, padding="same", dilation_rate=dilation_rate, 
               name=f"{prefix}_tcn_conv1d_{dilation_rate}")(x)
    x = layers.BatchNormalization(name=f"{prefix}_tcn_bn_{dilation_rate}")(x)
    x = layers.LeakyReLU(alpha=0.1, name=f"{prefix}_tcn_leaky_{dilation_rate}")(x)
    x = layers.Dropout(0.05, name=f"{prefix}_tcn_dropout_{dilation_rate}")(x)
    
    # Second dilated causal conv
    x = layers.Conv1D(filters, kernel_size, padding="same", dilation_rate=dilation_rate,
               name=f"{prefix}_tcn_conv1d_{dilation_rate}_2")(x)
    x = layers.BatchNormalization(name=f"{prefix}_tcn_bn_{dilation_rate}_2")(x)
    x = layers.LeakyReLU(alpha=0.1, name=f"{prefix}_tcn_leaky_{dilation_rate}_2")(x)
    x = layers.Dropout(0.05, name=f"{prefix}_tcn_dropout_{dilation_rate}_2")(x)
    
    # Add residual connection
    x = layers.Add(name=f"{prefix}_tcn_add_{dilation_rate}")([x, residual])
    
    return x

# Transformer Block
def Transformer_Block(x, embed_dim, num_heads=4, ff_dim=None, dropout=0.05, prefix=""):
    """
    Transformer Block with Multi-head Attention.
    
    Args:
        x (tf.Tensor): Input tensor
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward dimension
        dropout (float): Dropout rate
        prefix (str): Prefix for layer names
        
    Returns:
        tf.Tensor: Output tensor
    """
    if ff_dim is None:
        ff_dim = embed_dim * 4
        
    # Multi-head self attention
    residual = x
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_ln_1")(x)
    x = layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=embed_dim // num_heads,
        dropout=dropout,
        name=f"{prefix}_mha"
    )(x, x)
    x = layers.Dropout(dropout, name=f"{prefix}_mha_dropout")(x)
    x = layers.Add(name=f"{prefix}_add_1")([residual, x])
    
    # Feed Forward Network
    residual = x
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_ln_2")(x)
    x = layers.Dense(ff_dim, activation="relu", name=f"{prefix}_ff_1")(x)
    x = layers.Dropout(dropout, name=f"{prefix}_ff_dropout_1")(x)
    x = layers.Dense(embed_dim, name=f"{prefix}_ff_2")(x)
    x = layers.Dropout(dropout, name=f"{prefix}_ff_dropout_2")(x)
    x = layers.Add(name=f"{prefix}_add_2")([residual, x])
    
    return x

# Encoder Model
def Encoder(input_shape, out_dim=64, prefix=""):
    """
    Balanced Encoder with progressive dimensions.
    
    Args:
        input_shape (tuple): Input shape (window_width, n_features)
        out_dim (int): Output embedding dimension
        prefix (str): Prefix for layer names
        
    Returns:
        tuple: (input_layer, encoded_output)
    """
    input_layer = layers.Input(shape=input_shape, name=f"{prefix}_input")
    
    # Initial projection - keep small for initial feature extraction
    x = layers.Conv1D(16, 1, padding="same", name=f"{prefix}_initial_projection")(input_layer)
    x = layers.BatchNormalization(name=f"{prefix}_initial_bn")(x)
    x = layers.LeakyReLU(alpha=0.1, name=f"{prefix}_initial_leaky")(x)
    
    # ===== 3 TCN BLOCKS WITH PROGRESSIVE DIMENSIONS =====
    # First TCN Block (16 filters)
    x = TCN_Block(x, 16, dilation_rate=1, prefix=f"{prefix}_tcn1")
    x = layers.MaxPooling1D(pool_size=2, name=f"{prefix}_maxpool_1")(x)
    x = layers.Dropout(0.1, name=f"{prefix}_dropout_1")(x)
    
    # Second TCN Block (32 filters)
    x = TCN_Block(x, 32, dilation_rate=2, prefix=f"{prefix}_tcn2")
    x = layers.MaxPooling1D(pool_size=2, name=f"{prefix}_maxpool_2")(x)
    x = layers.Dropout(0.1, name=f"{prefix}_dropout_2")(x)
    
    # Third TCN Block (64 filters)
    x = TCN_Block(x, 64, dilation_rate=4, prefix=f"{prefix}_tcn3")
    x = layers.MaxPooling1D(pool_size=2, name=f"{prefix}_maxpool_3")(x)
    x = layers.Dropout(0.1, name=f"{prefix}_dropout_3")(x)
    
    # ===== 1 BiGRU (32 units) =====
    x = layers.Bidirectional(
        layers.GRU(32, return_sequences=True, name=f"{prefix}_bigru"),
        name=f"{prefix}_bidirectional"
    )(x)
    
    # ===== 1 TRANSFORMER (64 embedding dim) =====
    x = Transformer_Block(x, embed_dim=64, num_heads=2, prefix=f"{prefix}_transformer_final")
    
    # Flatten
    x = layers.Flatten(name=f"{prefix}_flatten")(x)
    
    # Dense Projection Layer - compress to out_dim dimensions
    x = layers.Dense(out_dim, name=f"{prefix}_dense")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn_final")(x)
    x = layers.LeakyReLU(alpha=0.1, name=f"{prefix}_leaky_final")(x)
    
    # L2 Normalize Features
    x = L2NormalizationLayer(axis=1, name=f"{prefix}_l2_norm")(x)
    
    return input_layer, x

def create_zeroshot_model(window_width=128, num_classes=18, embedding_dim=64):
    """
    Create a Zero-Shot HAR model with TCN and Transformer.
    
    Args:
        window_width (int): Window width
        num_classes (int): Number of classes
        embedding_dim (int): Embedding dimension
        
    Returns:
        tf.keras.Model: Zero-Shot model
    """
    # Define Input and Encoders with embedding_dim embedding dimension
    input_accel, encoded_accel = Encoder((window_width, 3), out_dim=embedding_dim, prefix="accel")
    input_gyro, encoded_gyro = Encoder((window_width, 3), out_dim=embedding_dim, prefix="gyro")

    # Compute Similarity Matrix (Batch-wise) using corrected cosine similarity
    similarity_matrix = SimilarityLayer(name="similarity_layer")([encoded_accel, encoded_gyro])

    # Merge encoded features for classifier
    merged_features = layers.Concatenate(name="merged_features")([encoded_accel, encoded_gyro])

    # Reshape for 1D convolution (batch_size, 128, 1) - Correct size for embedding_dim+embedding_dim=128
    x = layers.Reshape((embedding_dim*2, 1), name="classifier_reshape_for_cnn")(merged_features)

    # Two CNN blocks with progressive dimensions
    # First CNN block
    x = layers.Conv1D(64, kernel_size=3, padding='same', name="classifier_conv1")(x)
    x = layers.BatchNormalization(name="classifier_conv_bn1")(x)
    x = layers.LeakyReLU(alpha=0.1, name="classifier_conv_leaky1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="classifier_maxpool1")(x)
    x = layers.Dropout(0.05, name="classifier_conv_dropout1")(x)

    # Second CNN block
    x = layers.Conv1D(32, kernel_size=3, padding='same', name="classifier_conv2")(x)
    x = layers.BatchNormalization(name="classifier_conv_bn2")(x)
    x = layers.LeakyReLU(alpha=0.1, name="classifier_conv_leaky2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="classifier_maxpool2")(x)
    x = layers.Dropout(0.05, name="classifier_conv_dropout2")(x)

    # Flatten output
    x = layers.Flatten(name="classifier_flatten")(x)

    # Dense layers
    x = layers.Dense(64, name="classifier_dense1")(x)
    x = layers.BatchNormalization(name="classifier_dense_bn1")(x)
    x = layers.LeakyReLU(alpha=0.1, name="classifier_dense_leaky1")(x)
    x = layers.Dropout(0.05, name="classifier_dense_dropout1")(x)

    x = layers.Dense(32, name="classifier_dense2")(x)  # Reduced from 64 to 32
    x = layers.BatchNormalization(name="classifier_dense_bn2")(x)
    x = layers.LeakyReLU(alpha=0.1, name="classifier_dense_leaky2")(x)
    x = layers.Dropout(0.05, name="classifier_dense_dropout2")(x)

    # Output layer
    classifier_output = layers.Dense(num_classes, activation="softmax", name='classifier_output')(x)

    # Create the model
    model = Model(
        inputs=[input_accel, input_gyro],
        outputs=[similarity_matrix, classifier_output]
    )
    
    return model

def compute_contrastive_loss(similarity_matrix):
    """
    Compute supervised contrastive loss as per paper equation (15):
    L_scl = 1/2 * (L_CE(S, p) + L_CE(S^T, p))
    
    Contrastive Learning Explanation:
    - similarity_matrix[i,j] = cosine_similarity(accel_i, gyro_j)
    - p = [0, 1, 2, ..., N-1] are pseudo-labels (instance indices)
    - Goal: Make similarity_matrix[i,i] high (positive pairs) and similarity_matrix[i,j] low for i≠j (negative pairs)
    
    Cross-entropy with p[i] = i means:
    - We want similarity_matrix[i, p[i]] = similarity_matrix[i, i] to be the highest in row i
    - This encourages accel_i to be most similar to gyro_i (same sample = positive pair)
    - And less similar to gyro_j for j≠i (different samples = negative pairs)
    
    Args:
        similarity_matrix (tf.Tensor): Cosine similarity matrix S ∈ R^(N×N)
        
    Returns:
        tf.Tensor: Contrastive loss
    """
    batch_size = tf.shape(similarity_matrix)[0]
    
    # Create pseudo-labels p = [0, 1, ..., N-1] 
    # p[i] = i means "accel_i should be most similar to gyro_i"
    p = tf.range(batch_size, dtype=tf.int32)
    p_one_hot = tf.one_hot(p, depth=batch_size)
    
    # L_CE(S, p): Cross-entropy loss between similarity matrix and pseudo-labels
    # This makes similarity_matrix[i, i] the maximum in each row i
    # (accel_i most similar to gyro_i, less similar to other gyro_j)
    loss_1 = tf.keras.losses.categorical_crossentropy(p_one_hot, similarity_matrix, from_logits=True)
    
    # L_CE(S^T, p): Cross-entropy loss between transposed similarity matrix and pseudo-labels  
    # This makes similarity_matrix[i, i] the maximum in each column i
    # (gyro_i most similar to accel_i, less similar to other accel_j)
    similarity_matrix_t = tf.transpose(similarity_matrix)
    loss_2 = tf.keras.losses.categorical_crossentropy(p_one_hot, similarity_matrix_t, from_logits=True)
    
    # L_scl = 1/2 * (L_CE(S, p) + L_CE(S^T, p))
    # Symmetric contrastive loss for both directions
    contrastive_loss = 0.5 * (tf.reduce_mean(loss_1) + tf.reduce_mean(loss_2))
    
    return contrastive_loss

def compute_classification_loss(y_true, y_pred):
    """
    Compute classification loss as per paper equation (13):
    L_cls = -∑_{i=1}^C y_i log(ŷ_i)
    
    Args:
        y_true (tf.Tensor): True labels (one-hot encoded)
        y_pred (tf.Tensor): Predicted probabilities
        
    Returns:
        tf.Tensor: Classification loss
    """
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def compute_total_loss(similarity_matrix, y_true, y_pred, lambda_scl=1.0):
    """
    Compute total loss as per paper equation (16):
    L = L_cls + λ · L_scl
    
    Args:
        similarity_matrix (tf.Tensor): Cosine similarity matrix
        y_true (tf.Tensor): True labels (one-hot encoded)
        y_pred (tf.Tensor): Predicted probabilities  
        lambda_scl (float): Weight for contrastive loss
        
    Returns:
        tuple: (total_loss, classification_loss, contrastive_loss)
    """
    # Classification loss (L_cls)
    cls_loss = tf.reduce_mean(compute_classification_loss(y_true, y_pred))
    
    # Supervised contrastive loss (L_scl)
    scl_loss = compute_contrastive_loss(similarity_matrix)
    
    # Total loss (L = L_cls + λ · L_scl)
    total_loss = cls_loss + lambda_scl * scl_loss
    
    return total_loss, cls_loss, scl_loss

if __name__ == "__main__":
    # Test model creation
    model = create_zeroshot_model()
    
    # Print model architecture diagram
    tf.keras.utils.plot_model(model, "model.png", show_shapes=True)