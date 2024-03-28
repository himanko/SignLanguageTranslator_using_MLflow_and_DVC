
import tensorflow as tf
import numpy as np
import math
from tensorflow.python.keras.losses import categorical_crossentropy
from keras.callbacks import LearningRateScheduler, Callback
from tensorflow.python.keras import layers, optimizers
from SignLanguageTranslator.components.base_model import PrepareBaseModel
from SignLanguageTranslator.entity.config_entity import ModelTrainerConfig


class scce_with_ls:
    def __init__(self, y_true, y_pred, config: ModelTrainerConfig):
        self.config = config
        self.NUM_CLASSES = config.params_NUM_CLASSES
        # One Hot Encode Sparsely Encoded Target Sign
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, self.NUM_CLASSES, axis=1)
        y_true = tf.squeeze(y_true, axis=2)
        # Categorical Crossentropy with native label smoothing support
        return categorical_crossentropy(y_true, y_pred, label_smoothing=0.1)
    
class lrfn:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.lr_max = config.params_LR_MAX
        self.NUM_CLASSES = config.params_NUM_CLASSES
        self.num_warmup_steps = config.params_N_WARMUP_EPOCHS
    def lrfn(self, current_step, num_warmup_steps, lr_max, num_cycles=0.50):
        if current_step < num_warmup_steps:
            return lr_max * 2 ** -(num_warmup_steps - current_step)
        else:
            num_training_steps = self.NUM_CLASSES
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * self.lr_max
        
LR_SCHEDULE = [lrfn(step, num_warmup_steps=ModelTrainerConfig.params_N_WARMUP_EPOCHS, 
                    lr_max=ModelTrainerConfig.params_LR_MAX, num_cycles=0.50) for step in 
                    range(N_EPOCHS=ModelTrainerConfig.params_N_EPOCHS)]
# Learning Rate Callback
lr_callback = LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=1)

class WeightDecayCallback(Callback):
    def __init__(self, config: ModelTrainerConfig):
        self.config =config
        self.wd_ratio = config.params_WD_RATIO
        self.step_counter = 0
        self.model = PrepareBaseModel.get_base_model()
    
    def on_epoch_begin(self, epoch, logs=None):
        self.model.optimizer.weight_decay = self.model.optimizer.learning_rate * self.wd_ratio
        print(f'learning rate: {self.model.optimizer.learning_rate.numpy():.2e}, weight decay: {self.model.optimizer.weight_decay.numpy():.2e}')
