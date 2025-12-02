import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from salesRegressor.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    @staticmethod
    def rmspe_metric(y_true, y_pred):

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mask = y_true != 0
        return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))

    def evaluate(self):
        model = CatBoostRegressor()
        model.load_model(self.config.model_path)

        df_test = pd.read_csv(self.config.test_data_path, low_memory=False)

        X_test = df_test.drop(['Sales', 'Date'], axis=1)
        y_test = df_test['Sales']

        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)

        rmse = np.sqrt(mean_squared_error(np.expm1(y_test), y_pred))
        rmspe = self.rmspe_metric(np.expm1(y_test), y_pred)

        metrics = {
            "RMSE": float(rmse),
            "RMSPE": float(rmspe)
        }

        with open(self.config.metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)