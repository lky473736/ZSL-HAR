#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Architecture: Temporal Convolutional Network (TCN) + Transformer for HAR
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

# Custom Similarity Layer
class SimilarityLayer(layers.Layer):
    """Layer that calculates batch-wise similarity."""
    
    def call(self, inputs):
        return tf.matmul(inputs[0], inputs[1], transpose_b=True)

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

    # Compute Similarity Matrix (Batch-wise)
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

def create_embedding_model(model):
    """
    Create an embedding model from the trained model for feature extraction.
    
    Args:
        model (tf.keras.Model): Trained model
        
    Returns:
        tf.keras.Model: Embedding model
    """
    embedding_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer("merged_features").output]
    )
    return embedding_model

if __name__ == "__main__":
    # Test model creation
    model = create_zeroshot_model()
    
    # Print model architecture diagram
    tf.keras.utils.plot_model(model, "model.png", show_shapes=True)