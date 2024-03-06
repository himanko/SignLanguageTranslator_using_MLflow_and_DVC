from SignLanguageTranslator import logger
from SignLanguageTranslator.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from SignLanguageTranslator.pipeline.stage_02_landmarks_extraction import LandmarksExtractionPipeline

# STAGE_NAME ="Data Ingestion Stage"

# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     obj = DataIngestionTrainingPipeline()
#     obj.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e


STAGE_NAME = "Landmarks Extraction stage"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_landmark_extraction = LandmarksExtractionPipeline()
   prepare_landmark_extraction.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e