from salesRegressor.config.configuration import ConfigurationManager
from salesRegressor.components.model_eval import ModelEvaluation
from salesRegressor import logger

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluator = ModelEvaluation(model_evaluation_config)
        model_evaluator.evaluate()