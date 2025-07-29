#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Architecture: Temporal Convolutional Network (TCN) + BiGRU + Transformer for HAR
Implements the zero-shot learning architecture for activity recognition.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

# Custom L2Normalization layer with manual implementation
class L2NormalizationLayer(layers.Layer):
    """Layer that normalizes inputs to unit L2 norm manually implementing the math."""
    
    def __init__(self, axis=1, **kwargs):
        super(L2NormalizationLayer, self).__init__(**kwargs)
        self.axis = axis
        
    def call(self, inputs):
        # Manual L2 normalization: x / ||x||_2
        # ||x||_2 = sqrt(sum(x_i^2))
        squared_sum = tf.reduce_sum(tf.square(inputs), axis=self.axis, keepdims=True)
        l2_norm = tf.sqrt(squared_sum + 1e-12)  # Add epsilon to avoid division by zero
        normalized = inputs / l2_norm
        return normalized

# Custom Cosine Similarity Layer implementing paper equations (2) and (14)
class SimilarityLayer(layers.Layer):
    """
    Layer that calculates cosine similarity matrix implementing paper equations manually:
    Equation (2): sim(f^(a), f^(g)) = (f^(a) · f^(g)) / (||f^(a)||_2 ||f^(g)||_2)
    Equation (14): S_{i,j} = (f^(a)_i · f^(g)_j) / (||f^(a)_i|| · ||f^(g)_j||)
    """
    
    def call(self, inputs):
        f_accel, f_gyro = inputs
        
        # Manual cosine similarity computation instead of just matmul
        # Step 1: Compute dot products f^(a)_i · f^(g)_j for all pairs
        dot_products = tf.matmul(f_accel, f_gyro, transpose_b=True)
        
        # Step 2: Compute L2 norms manually
        # ||f^(a)_i||_2 = sqrt(sum(f^(a)_i^2))
        accel_norms = tf.sqrt(tf.reduce_sum(tf.square(f_accel), axis=1, keepdims=True) + 1e-12)
        gyro_norms = tf.sqrt(tf.reduce_sum(tf.square(f_gyro), axis=1, keepdims=True) + 1e-12)
        
        # Step 3: Compute denominator ||f^(a)_i|| * ||f^(g)_j|| for all pairs
        norm_products = tf.matmul(accel_norms, gyro_norms, transpose_b=True)
        
        # Step 4: Final cosine similarity S_{i,j} = dot_product / norm_product
        similarity_matrix = dot_products / norm_products
        
        return similarity_matrix

# TCN Block
def TCN_Block(x, filters, kernel_size=3, dilation_rate=1, prefix=""):
    """Temporal Convolutional Network Block."""
    residual = x
    
    if int(x.shape[-1]) != filters:
        residual = layers.Conv1D(filters, 1, padding="same", name=f"{prefix}_residual_conv")(x)
    
    x = layers.Conv1D(filters, kernel_size, padding="same", dilation_rate=dilation_rate, 
               name=f"{prefix}_tcn_conv1d_{dilation_rate}")(x)
    x = layers.BatchNormalization(name=f"{prefix}_tcn_bn_{dilation_rate}")(x)
    x = layers.LeakyReLU(alpha=0.1, name=f"{prefix}_tcn_leaky_{dilation_rate}")(x)
    x = layers.Dropout(0.05, name=f"{prefix}_tcn_dropout_{dilation_rate}")(x)
    
    x = layers.Conv1D(filters, kernel_size, padding="same", dilation_rate=dilation_rate,
               name=f"{prefix}_tcn_conv1d_{dilation_rate}_2")(x)
    x = layers.BatchNormalization(name=f"{prefix}_tcn_bn_{dilation_rate}_2")(x)
    x = layers.LeakyReLU(alpha=0.1, name=f"{prefix}_tcn_leaky_{dilation_rate}_2")(x)
    x = layers.Dropout(0.05, name=f"{prefix}_tcn_dropout_{dilation_rate}_2")(x)
    
    x = layers.Add(name=f"{prefix}_tcn_add_{dilation_rate}")([x, residual])
    
    return x

# Transformer Block
def Transformer_Block(x, embed_dim, num_heads=2, ff_dim=None, dropout=0.05, prefix=""):
    """Transformer Block with Multi-head Attention."""
    if ff_dim is None:
        ff_dim = embed_dim * 2  # Reduced from 4x to 2x
        
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
    
    residual = x
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_ln_2")(x)
    x = layers.Dense(ff_dim, activation="relu", name=f"{prefix}_ff_1")(x)
    x = layers.Dropout(dropout, name=f"{prefix}_ff_dropout_1")(x)
    x = layers.Dense(embed_dim, name=f"{prefix}_ff_2")(x)
    x = layers.Dropout(dropout, name=f"{prefix}_ff_dropout_2")(x)
    x = layers.Add(name=f"{prefix}_add_2")([residual, x])
    
    return x

# Encoder Model
def Encoder(input_shape, out_dim=24, prefix=""):  # Further reduced from 32 to 24
    """Encoder with TCN + BiGRU + Transformer architecture."""
    input_layer = layers.Input(shape=input_shape, name=f"{prefix}_input")
    
    x = layers.Conv1D(6, 1, padding="same", name=f"{prefix}_initial_projection")(input_layer)  # Further reduced from 8 to 6
    x = layers.BatchNormalization(name=f"{prefix}_initial_bn")(x)
    x = layers.LeakyReLU(alpha=0.1, name=f"{prefix}_initial_leaky")(x)
    
    # 3 TCN Blocks - further reduced filters
    x = TCN_Block(x, 6, dilation_rate=1, prefix=f"{prefix}_tcn1")  # Further reduced from 8 to 6
    x = layers.MaxPooling1D(pool_size=2, name=f"{prefix}_maxpool_1")(x)
    x = layers.Dropout(0.1, name=f"{prefix}_dropout_1")(x)
    
    x = TCN_Block(x, 12, dilation_rate=2, prefix=f"{prefix}_tcn2")  # Further reduced from 16 to 12
    x = layers.MaxPooling1D(pool_size=2, name=f"{prefix}_maxpool_2")(x)
    x = layers.Dropout(0.1, name=f"{prefix}_dropout_2")(x)
    
    x = TCN_Block(x, 24, dilation_rate=4, prefix=f"{prefix}_tcn3")  # Further reduced from 32 to 24
    x = layers.MaxPooling1D(pool_size=2, name=f"{prefix}_maxpool_3")(x)
    x = layers.Dropout(0.1, name=f"{prefix}_dropout_3")(x)
    
    # BiGRU - further reduced units
    x = layers.Bidirectional(
        layers.GRU(12, return_sequences=True, name=f"{prefix}_bigru"),  # Further reduced from 16 to 12
        name=f"{prefix}_bidirectional"
    )(x)
    
    # Transformer - further reduced embed_dim
    x = Transformer_Block(x, embed_dim=24, num_heads=2, prefix=f"{prefix}_transformer_final")  # Further reduced from 32 to 24
    
    # Flatten and Dense
    x = layers.Flatten(name=f"{prefix}_flatten")(x)
    x = layers.Dense(out_dim, name=f"{prefix}_dense")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn_final")(x)
    x = layers.LeakyReLU(alpha=0.1, name=f"{prefix}_leaky_final")(x)
    
    # Manual L2 Normalize instead of tf.math.l2_normalize
    x = L2NormalizationLayer(axis=1, name=f"{prefix}_l2_norm")(x)
    
    return input_layer, x

def create_zeroshot_model(window_width=128, num_classes=18, embedding_dim=24):  # Further reduced from 32 to 24
    """Create a Zero-Shot HAR model with TCN and Transformer."""
    # Define Input and Encoders
    input_accel, encoded_accel = Encoder((window_width, 3), out_dim=embedding_dim, prefix="accel")
    input_gyro, encoded_gyro = Encoder((window_width, 3), out_dim=embedding_dim, prefix="gyro")

    # Compute Similarity Matrix using manual cosine similarity
    similarity_matrix = SimilarityLayer(name="similarity_layer")([encoded_accel, encoded_gyro])

    # Merge encoded features for classifier
    merged_features = layers.Concatenate(name="merged_features")([encoded_accel, encoded_gyro])

    # Classification head
    x = layers.Reshape((embedding_dim*2, 1), name="classifier_reshape_for_cnn")(merged_features)

    # Two CNN blocks - further reduced filters
    x = layers.Conv1D(24, kernel_size=3, padding='same', name="classifier_conv1")(x)  # Further reduced from 32 to 24
    x = layers.BatchNormalization(name="classifier_conv_bn1")(x)
    x = layers.LeakyReLU(alpha=0.1, name="classifier_conv_leaky1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="classifier_maxpool1")(x)
    x = layers.Dropout(0.05, name="classifier_conv_dropout1")(x)

    x = layers.Conv1D(12, kernel_size=3, padding='same', name="classifier_conv2")(x)  # Further reduced from 16 to 12
    x = layers.BatchNormalization(name="classifier_conv_bn2")(x)
    x = layers.LeakyReLU(alpha=0.1, name="classifier_conv_leaky2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="classifier_maxpool2")(x)
    x = layers.Dropout(0.05, name="classifier_conv_dropout2")(x)

    x = layers.Flatten(name="classifier_flatten")(x)

    # Dense layers - further reduced units
    x = layers.Dense(24, name="classifier_dense1")(x)  # Further reduced from 32 to 24
    x = layers.BatchNormalization(name="classifier_dense_bn1")(x)
    x = layers.LeakyReLU(alpha=0.1, name="classifier_dense_leaky1")(x)
    x = layers.Dropout(0.05, name="classifier_dense_dropout1")(x)

    x = layers.Dense(12, name="classifier_dense2")(x)  # Further reduced from 16 to 12
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
    Manual implementation of supervised contrastive loss (paper equation 15):
    L_scl = 1/2 * (L_CE(S, p) + L_CE(S^T, p))
    
    Instead of using tf.keras.losses.categorical_crossentropy, implement manually
    """
    batch_size = tf.shape(similarity_matrix)[0]
    
    # Create pseudo-labels p = [0, 1, ..., N-1] 
    p = tf.range(batch_size, dtype=tf.int32)
    p_one_hot = tf.one_hot(p, depth=batch_size)
    
    # Manual cross-entropy computation instead of tf.keras.losses.categorical_crossentropy
    # L_CE(S, p): -∑_j p_j log(softmax(S_i))
    
    # Compute softmax manually
    softmax_sim = tf.exp(similarity_matrix) / tf.reduce_sum(tf.exp(similarity_matrix), axis=1, keepdims=True)
    
    # Manual cross-entropy: -∑ y_true * log(y_pred)
    epsilon = 1e-12
    softmax_sim_clipped = tf.clip_by_value(softmax_sim, epsilon, 1.0 - epsilon)
    loss_1 = -tf.reduce_sum(p_one_hot * tf.math.log(softmax_sim_clipped), axis=1)
    loss_1 = tf.reduce_mean(loss_1)
    
    # L_CE(S^T, p): Same for transposed matrix
    similarity_matrix_t = tf.transpose(similarity_matrix)
    softmax_sim_t = tf.exp(similarity_matrix_t) / tf.reduce_sum(tf.exp(similarity_matrix_t), axis=1, keepdims=True)
    softmax_sim_t_clipped = tf.clip_by_value(softmax_sim_t, epsilon, 1.0 - epsilon)
    loss_2 = -tf.reduce_sum(p_one_hot * tf.math.log(softmax_sim_t_clipped), axis=1)
    loss_2 = tf.reduce_mean(loss_2)
    
    # L_scl = 1/2 * (L_CE(S, p) + L_CE(S^T, p))
    contrastive_loss = 0.5 * (loss_1 + loss_2)
    
    return contrastive_loss

def compute_classification_loss(y_true, y_pred):
    """
    Manual implementation of classification loss (paper equation 13):
    L_cls = -∑_{i=1}^C y_i log(ŷ_i)
    
    Instead of using tf.keras.losses.categorical_crossentropy, implement manually
    """
    epsilon = 1e-12
    # Clip predictions to prevent log(0)
    y_pred_clipped = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Manual cross-entropy: -∑_{i=1}^C y_i log(ŷ_i)
    cross_entropy = -tf.reduce_sum(y_true * tf.math.log(y_pred_clipped), axis=1)
    
    # Return mean loss across batch
    return tf.reduce_mean(cross_entropy)

def compute_total_loss(similarity_matrix, y_true, y_pred, lambda_scl=1.0):
    """
    Manual implementation of total loss (paper equation 16):
    L = L_cls + λ · L_scl
    """
    # Classification loss (L_cls) - manual implementation
    cls_loss = compute_classification_loss(y_true, y_pred)
    
    # Supervised contrastive loss (L_scl) - manual implementation
    scl_loss = compute_contrastive_loss(similarity_matrix)
    
    # Total loss (L = L_cls + λ · L_scl)
    total_loss = cls_loss + lambda_scl * scl_loss
    
    return total_loss, cls_loss, scl_loss

if __name__ == "__main__":
    # Test model creation
    model = create_zeroshot_model()
    
    # Print model architecture diagram
    tf.keras.utils.plot_model(model, "model.png", show_shapes=True)