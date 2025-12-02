from salesRegressor.constants import *
from salesRegressor.utils.common import read_yaml, create_directories
from salesRegressor.entity.config_entity import (DataIngestionConfig, DataValidationConfig,
                                                 DataTransformationConfig, ModelTrainerConfig,
                                                 ModelEvaluationConfig)


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
    
    def get_data_validation_config(self) -> DataValidationConfig:
        
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
        )
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        
        dt = self.config.data_transformation
        
        create_directories([dt.root_dir])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=Path(dt.root_dir),
            sales_file=Path(dt.sales_file),
            store_file=Path(dt.store_file),
            cleaned_data_file=Path(dt.cleaned_data_file),
            train_file=Path(dt.train_file),
            test_file=Path(dt.test_file),
            test_size=float(dt.test_size)
            )
        
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.CatBoostParams

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_file=config.train_file,
            test_file=config.test_file,
            model_file=config.model_file,
            iterations=params.iterations,
            learning_rate=params.learning_rate,
            depth=params.depth,
            loss_function=params.loss_function,
            early_stopping_rounds=params.early_stopping_rounds,
            verbose=params.verbose
            )
        
        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        
        create_directories([config.root_dir])
        
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            model_path=config.model_path,
            test_data_path=config.test_data_path,
            metrics_file=config.metrics_file
            )

        return model_evaluation_config