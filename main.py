from SignLanguageTranslator import logger
from SignLanguageTranslator.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from SignLanguageTranslator.pipeline.stage_02_landmarks_extraction import LandmarksExtractionPipeline
from SignLanguageTranslator.pipeline.stage_03_preprocessing import PreprocessingPipeline
from SignLanguageTranslator.pipeline.stage_04_prepare_base_model import BaseModelPipeline
from SignLanguageTranslator.pipeline.stage_05_model_training import TrainingPipeline

# STAGE_NAME ="Data Ingestion Stage"

# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     obj = DataIngestionTrainingPipeline()
#     obj.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e


# STAGE_NAME = "Landmarks Extraction stage"
# try: 
#    logger.info(f"*******************")
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#    prepare_landmark_extraction = LandmarksExtractionPipeline()
#    prepare_landmark_extraction.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e


# STAGE_NAME = "Preprocessing stage"
# try: 
#    logger.info(f"*******************")
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#    prepare_landmark_extraction = PreprocessingPipeline()
#    prepare_landmark_extraction.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e


STAGE_NAME = "Prepare Base Model stage"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_landmark_extraction = BaseModelPipeline()
   prepare_landmark_extraction.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Training stage"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_landmark_extraction = TrainingPipeline()
   prepare_landmark_extraction.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
