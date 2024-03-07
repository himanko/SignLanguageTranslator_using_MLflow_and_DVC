from SignLanguageTranslator.config.configuration import ConfigurationManager
from SignLanguageTranslator.components.preprocessing import Preprocessing
from SignLanguageTranslator import logger

STAGE_NAME = "Preprocessing stage"

class PreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Initialize configuration manager
            config = ConfigurationManager()
            # Get landmarks extraction configuratio
            preprocessing_config = config.get_preprocessing_config()
            # Initialize LandmarksExtraction with the configuration
            preprocessing = Preprocessing(config=preprocessing_config)
            # Call the apply method to process videos
            preprocessing.apply()

        except Exception as e:
            logger.exception(f"Error in PreprocessingPipeline: {e}")
            raise e
        




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PreprocessingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e