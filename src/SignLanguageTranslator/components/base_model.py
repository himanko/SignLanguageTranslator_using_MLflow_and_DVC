import os 
import numpy as np
import pandas as pd
import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers.core import Reshape, Dropout, Dense
from tensorflow.python.keras.activations import softmax, gelu
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.keras.models import sequential
from SignLanguageTranslator.utils.base_model_utills import *
from SignLanguageTranslator.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path
from SignLanguageTranslator import logger

class PrepareBaseModel:
    def __init__(self, config:PrepareBaseModelConfig):
        self.config = config

    @staticmethod
    def save_model_diagram(path: Path, model):
        plot_model(model, show_shapes=True, to_file=path)


    def get_base_model(self):
        try:
            inputs = InputLayer(shape=(30, 104, 3))

            # inputs = augment_fn(inputs_l, max_len=30)
            lips = tf.slice(inputs, [0, 0, 0, 0], [-1, 35, 40, 3])
            lh = tf.slice(inputs, [0, 0, 40, 0], [-1, 35, 21, 3])
            po = tf.slice(inputs, [0, 0, 61, 0], [-1, 35, 22, 3])
            rh = tf.slice(inputs, [0, 0, 83, 0], [-1, 35, 21, 3])

            lips = Reshape((35, 40*3))(lips)
            lh = Reshape((35, 21*3))(lh)
            po = Reshape((35, 22*3))(po)
            rh = Reshape((35, 21*3))(rh)
        

            embedding_units = [256, 256] # tune this

            # dense encoder model
            lips = Dense(512, activation=gelu)(lips)
            lh = Dense(512, activation=gelu)(lh)
            po = Dense(512, activation=gelu)(po)
            rh = Dense(512, activation=gelu)(rh)
            
            x = tf.concat((lips, lh, po, rh), axis=2)
            # x = tf.reduce_mean(x, axis=3)
            for n in embedding_units:
                x = dense_block(n)(x)
            x = Transformer(num_blocks=4)(x)
            x = tf.reduce_sum(x, axis=1)
            dense = Dense(256, activation=gelu)(x)
            drop = Dropout(0.1)(x)

            out = Dense(250, activation=softmax, name="outputs")(x)
            
            model = tf.keras.Model(inputs=inputs, outputs=out)
                
            return model
        
        except Exception as e:
            raise e
    def model_summary(self):
        base_model = self.get_base_model()
        base_model.summary()

    def apply(self):
        base_model = self.get_base_model()
        self.save_model_diagram(
            path=self.config.saved_model_diagram,
            model=base_model
        )



