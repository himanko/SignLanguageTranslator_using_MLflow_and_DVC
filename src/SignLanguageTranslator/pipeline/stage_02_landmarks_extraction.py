from SignLanguageTranslator.config.configuration import ConfigurationManager
from SignLanguageTranslator.components.landmarks_extraction import LandmarksExtraction
from SignLanguageTranslator import logger


STAGE_NAME = "Landmarks Extraction stage"

class LandmarksExtractionPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Initialize configuration manager
            config = ConfigurationManager()
            # Get landmarks extraction configuratio
            landmarks_extraction_config = config.get_landmarks_extraction_config()
            # Initialize LandmarksExtraction with the configuration
            landmarks_extraction = LandmarksExtraction(config=landmarks_extraction_config)
            # Call the apply method to process videos
            landmarks_extraction.apply()

        except Exception as e:
            logger.exception(f"Error in LandmarksExtractionPipeline: {e}")
            raise e

        
        




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = LandmarksExtractionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e



