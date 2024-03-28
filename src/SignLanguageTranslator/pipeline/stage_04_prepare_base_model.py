from SignLanguageTranslator.config.configuration import ConfigurationManager
from SignLanguageTranslator.components.base_model import PrepareBaseModel
from SignLanguageTranslator import logger


STAGE_NAME = "Prepare Base Model stage"

class BaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Initialize configuration manager
            config = ConfigurationManager()
            # Get Base Model configuratio
            base_model_config = config.get_base_model_config()
            # Initialize LandmarksExtraction with the configuration
            prepare_base_model = PrepareBaseModel(config=base_model_config)
            prepare_base_model.get_base_model()
            prepare_base_model.model_summary()
            prepare_base_model.apply()


        except Exception as e:
            logger.exception(f"Error in BaseModelPipeline: {e}")
            raise e
        




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = BaseModelPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e