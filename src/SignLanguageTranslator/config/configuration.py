from SignLanguageTranslator.constants import *
from SignLanguageTranslator.utils.common import read_yaml, create_directories
from SignLanguageTranslator.entity.config_entity import DataIngestionConfig, LandmarksExtractionConfig, PreprocessingConfig

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