from dataclasses import dataclass
from pathlib import Path


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
    params_UNITS: int
    params_NUM_BLOCKS: int
    params_MLP_RATIO: int
    params_EMBEDDING_DROPOUT: float
    params_MLP_DROPOUT_RATIO: float
    params_CLASSIFIER_DROPOUT_RATIO: float
    params_N_EPOCHS: int
    params_LR_MAX: int
    params_N_WARMUP_EPOCHS: float
    params_WD_RATIO: float
    params_NUM_CLASSES: int