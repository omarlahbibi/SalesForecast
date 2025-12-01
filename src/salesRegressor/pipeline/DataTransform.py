from salesRegressor.config.configuration import ConfigurationManager
from salesRegressor.components.data_transform import DataTransformation
from salesRegressor import logger


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)

        sales_df, store_df = data_transformation._load_data()

        sales_df = data_transformation._clean_sales(sales_df)
        store_df = data_transformation._clean_store(store_df)

        merged_df = data_transformation._merge(sales_df, store_df)

        merged_df = data_transformation._add_time_features(merged_df)

        merged_df = data_transformation._feature_engineering(merged_df)

        merged_df = data_transformation._log_transform(
            merged_df, skewed_features=["Sales", "Customers", "CompetitionDistance"])

        merged_df.to_csv(data_transformation_config.cleaned_data_file, index=False)
        logger.info(f"Saving cleaned merged dataset")

        train_df, test_df = data_transformation._train_test_split(merged_df)
        train_df.to_csv(data_transformation_config.train_file, index=False)
        test_df.to_csv(data_transformation_config.test_file, index=False)

        logger.info("Data transformation completed successfully.")