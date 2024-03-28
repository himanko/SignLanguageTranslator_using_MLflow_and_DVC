from SignLanguageTranslator.config.configuration import ConfigurationManager
from SignLanguageTranslator.components.model_trainer import Training
from SignLanguageTranslator import logger


STAGE_NAME = "Training stage"

class TrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Initialize configuration manager
            config = ConfigurationManager()
            # Get Base Model configuratio
            training_config = config.get_model_trainer_config()
            # Initialize LandmarksExtraction with the configuration
            prepare_base_model = Training(config=training_config)
            prepare_base_model.train()
            


        except Exception as e:
            logger.exception(f"Error in TrainingPipeline: {e}")
            raise e
        




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e