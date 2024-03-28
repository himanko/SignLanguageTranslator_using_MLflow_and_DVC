import os 
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.python.keras.layers.core import Dropout, Dense
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers.pooling import MaxPool1D, GlobalAvgPool1D
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.activations import softmax, gelu
from tensorflow.python.keras.initializers.initializers_v2 import HeUniform, GlorotUniform, Constant
from tensorflow.python.keras import layers, optimizers
from SignLanguageTranslator.entity.config_entity import PrepareBaseModelConfig
from sklearn.model_selection import train_test_split
from SignLanguageTranslator import logger


#activations
GELU = gelu

# Initiailizers
INIT_HE_UNIFORM = HeUniform
INIT_GLOROT_UNIFORM = GlorotUniform
INIT_ZEROS = Constant(0.0)


features = np.load("/kaggle/working/feature_data.npy")
labels = np.load("/kaggle/working/feature_labels.npy")

train_x, val_x, train_y, val_y = train_test_split(features, labels, test_size=0.20, random_state=42)


def scaled_dot_product(q, k, v, softmax):
    try:
        # Calculates Q . K(transpose)
        qkt = tf.matmul(q, k, transpose_b=True)
        # Calculates scaling factor
        dk = tf.math.sqrt(tf.cast(q.shape[-1], dtype=tf.float32))
        scaled_qkt = qkt / dk
        softmax_output = softmax(scaled_qkt)

        z = tf.matmul(softmax_output, v)
        # Shape: (m, Tx, depth), same shape as q, k, v
        return z
    
    except Exception as e:
        raise e


class MultiHeadAttention(tf.keras.layers.Layer):
        def __init__(self, d_model, num_of_heads):
            super(MultiHeadAttention, self).__init__()
            self.d_model = d_model
            self.num_of_heads = num_of_heads
            self.depth = d_model // num_of_heads
            self.wq = [Dense(self.depth) for i in range(num_of_heads)]
            self.wk = [Dense(self.depth) for i in range(num_of_heads)]
            self.wv = [Dense(self.depth) for i in range(num_of_heads)]
            self.wo = Dense(d_model)
            self.softmax = softmax()

        def call(self, x):
            multi_attn = []
            for i in range(self.num_of_heads):
                Q = self.wq[i](x)
                K = self.wk[i](x)
                V = self.wv[i](x)
                multi_attn.append(scaled_dot_product(Q, K, V, self.softmax))
            multi_head = tf.concat(multi_attn, axis=-1)
            multi_head_attention = self.wo(multi_head)
            return multi_head_attention
    

class Transformer(tf.keras.Model):     
    def __init__(self, num_blocks, config: PrepareBaseModelConfig):
        super(Transformer, self).__init__(name='transformer')
        self.num_blocks = num_blocks
        self.config = config
        self.UNITS = config.params_UNITS
        self.mhas = []
        self.mlps = []

        for i in range(self.num_blocks):
            # Multi Head Attention
            self.mhas.append(MultiHeadAttention(self.UNITS, 8))  # Modify the arguments as needed
            # Multi Layer Perception
            self.mlps.append(sequential([
                Dense(self.UNITS, activation=GELU, kernel_initializer=INIT_GLOROT_UNIFORM),
                Dropout(0.30),
                Dense(self.UNITS, kernel_initializer=INIT_HE_UNIFORM),
            ]))

    def call(self, x):
        for mha, mlp in zip(self.mhas, self.mlps):
            x = x + mha(x)
            x = x + mlp(x)

        return x
    
def dense_block(units):
    try:
        fc = layers.Dense(units)
        norm = layers.LayerNormalization()
        act = layers.Activation(gelu)
        drop = layers.Dropout(0.05)
        return lambda x: drop(act(norm(fc(x))))
    
    except Exception as e:
        raise e