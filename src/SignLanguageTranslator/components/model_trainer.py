import os 
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tensorflow import keras
import tensorflow as tf
from keras.optimizers import AdamW
from keras.metrics import Accuracy, sparse_top_k_categorical_accuracy
from SignLanguageTranslator.utils.base_model_utills import *
from SignLanguageTranslator.entity.config_entity import ModelTrainerConfig
from SignLanguageTranslator.utils.model_trainer_utills import *

from SignLanguageTranslator.components.base_model import PrepareBaseModel

from SignLanguageTranslator import logger



class Training:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.model = None  # Initialize model attribute

    def get_base_model(self):
        prepare_base_model = PrepareBaseModel(self.config)  
        self.model = prepare_base_model.get_base_model()  # Store the model

    @staticmethod
    def save_model(path: Path, model):
        model.save(path)

    def train(self):
        self.scce_with_ls = scce_with_ls()
        self.lr_callback = lr_callback
        self.WeightDecayCallback = WeightDecayCallback()

        self.optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-6, clipnorm=0.05)

        self.model.compile(optimizer=self.optimizer,
              loss=self.scce_with_ls,
              metrics=[Accuracy, sparse_top_k_categorical_accuracy])

        self.model.fit(
            train_x, train_y, epochs=self.config.params_N_EPOCHS, 
            validation_data=(val_x, val_y), batch_size=self.config.params_batch_size,
            callbacks=[
                lr_callback,
                WeightDecayCallback()
                ]
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )


