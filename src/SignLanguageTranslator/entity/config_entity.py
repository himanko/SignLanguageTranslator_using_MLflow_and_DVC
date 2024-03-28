from dataclasses import dataclass
from pathlib import Path
import tensorflow as tf


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class LandmarksExtractionConfig:
    root_dir: Path
    com_dir: Path


@dataclass(frozen=True)
class PreprocessingConfig:
    root_dir: Path
    com_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    saved_model_diagram_path: Path
    params_UNITS: tf.int32
    params_NUM_BLOCKS: tf.int32
    params_MLP_RATIO: tf.int32
    params_EMBEDDING_DROPOUT: tf.float32
    params_MLP_DROPOUT_RATIO: tf.float32
    params_CLASSIFIER_DROPOUT_RATIO: tf.float32
    params_N_EPOCHS: tf.int32
    params_LR_MAX: tf.int32
    params_N_WARMUP_EPOCHS: tf.float32
    params_WD_RATIO: tf.float32
    params_NUM_CLASSES: tf.int32


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    trained_model_path: Path
    params_UNITS: tf.int32
    params_NUM_BLOCKS: tf.int32
    params_MLP_RATIO: tf.int32
    params_EMBEDDING_DROPOUT: tf.float32
    params_MLP_DROPOUT_RATIO: tf.float32
    params_CLASSIFIER_DROPOUT_RATIO: tf.float32
    params_N_EPOCHS: tf.int32
    params_batch_size: tf.int32
    params_LR_MAX: tf.int32
    params_N_WARMUP_EPOCHS: tf.float32
    params_WD_RATIO: tf.float32
    params_NUM_CLASSES: tf.int32