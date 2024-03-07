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