from SignLanguageTranslator.constants import *
from SignLanguageTranslator.utils.common import read_yaml, create_directories
from SignLanguageTranslator.entity.config_entity import (DataIngestionConfig, 
                                                         LandmarksExtractionConfig, 
                                                         PreprocessingConfig, 
                                                         PrepareBaseModelConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config

    
    
    def get_landmarks_extraction_config(self) -> LandmarksExtractionConfig:
        config = self.config.landmarks_extraction

        create_directories([config.root_dir])

        landmarks_extraction_config = LandmarksExtractionConfig(
            root_dir=config.root_dir,
            com_dir=config.com_dir 
        )

        return landmarks_extraction_config
    
    def get_preprocessing_config(self) -> PreprocessingConfig:
        config = self.config.preprocessing

        create_directories([config.root_dir])

        preprocessing_config = PreprocessingConfig(
            root_dir=config.root_dir,
            com_dir=config.com_dir 
        )

        return preprocessing_config
    
    def get_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.get_base_model

        create_directories([config.root_dir])

        base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            saved_model_diagram_path=Path(config.saved_model_diagram_path),
            params_UNITS=self.params.UNITS,
            params_NUM_BLOCKS=self.params.NUM_BLOCKS,
            params_MLP_RATIO=self.params.MLP_RATIO,
            params_EMBEDDING_DROPOUT=self.params.EMBEDDING_DROPOUT,
            params_MLP_DROPOUT_RATIO=self.params.MLP_DROPOUT_RATIO,
            params_CLASSIFIER_DROPOUT_RATIO=self.params.CLASSIFIER_DROPOUT_RATIO,
            params_N_EPOCHS=self.params.N_EPOCHS,
            params_LR_MAX=self.params.LR_MAX,
            params_N_WARMUP_EPOCHS=self.params.N_WARMUP_EPOCHS,
            params_WD_RATIO=self.params.WD_RATIO,
            params_NUM_CLASSES=self.params.NUM_CLASSES

        )

        return base_model_config