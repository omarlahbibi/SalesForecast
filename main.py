from salesRegressor import logger
from salesRegressor.pipeline.DataIngest import DataIngestionTrainingPipeline
from salesRegressor.pipeline.DataVal import DataValidationTrainingPipeline
from salesRegressor.pipeline.DataTransform import DataTransformationTrainingPipeline
from salesRegressor.pipeline.ModelTrainer import ModelTrainerTrainingPipeline
from salesRegressor.pipeline.ModelEval import ModelEvaluationTrainingPipeline


STAGE_NAME = "Data Ingestion stage"

try:
   logger.info(f">>>>>> {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Validation stage"

try:
   logger.info(f">>>>>> {STAGE_NAME} started <<<<<<") 
   data_validation = DataValidationTrainingPipeline()
   data_validation.main()
   logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Transformation stage"

try:
   logger.info(f">>>>>> {STAGE_NAME} started <<<<<<") 
   data_transformation = DataTransformationTrainingPipeline()
   data_transformation.main()
   logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Trainer stage"

try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
   model_trainer = ModelTrainerTrainingPipeline()
   model_trainer.main()
   logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Evaluation stage"

try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
   model_evaluation = ModelEvaluationTrainingPipeline()
   model_evaluation.main()
   logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e