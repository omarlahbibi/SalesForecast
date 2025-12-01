import pandas as pd
from catboost import CatBoostRegressor, Pool
from salesRegressor import logger
from salesRegressor.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):

        train_df = pd.read_csv(self.config.train_file, low_memory=False)
        test_df = pd.read_csv(self.config.test_file, low_memory=False)

        y_train = train_df['Sales']
        X_train = train_df.drop(['Sales', 'Date'], axis=1)

        y_test = test_df['Sales']
        X_test = test_df.drop(['Sales', 'Date'], axis=1)

        cat_features = [col for col in X_train.columns if X_train[col].dtype == 'object' or "StoreType" in col 
                        or "Assortment" in col]

        logger.info(f"Categorical features: {cat_features}")

        train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
        test_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)

        model = CatBoostRegressor(
            iterations=self.config.iterations,
            learning_rate=self.config.learning_rate,
            depth=self.config.depth,
            loss_function=self.config.loss_function,
            verbose=self.config.verbose
        )

        logger.info("Training CatBoost model...")
        model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=self.config.early_stopping_rounds)

        model.save_model(self.config.model_file)
        logger.info(f"Model saved to: {self.config.model_file}")

        return model