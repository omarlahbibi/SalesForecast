import os
from salesRegressor import logger
from salesRegressor.entity.config_entity import DataValidationConfig


class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config
    
    def validate_all_files_exist(self) -> bool:
        
        try:
            data_dir = os.path.join("artifacts", "data_ingestion", "rossmann-store-sales")
            all_files = os.listdir(data_dir)

            missing_files = [
                file for file in self.config.ALL_REQUIRED_FILES
                if file not in all_files
                ]

            validation_status = (len(missing_files) == 0)

            with open(self.config.STATUS_FILE, 'w') as f:
                if validation_status:
                    f.write("Validation status: True\nAll required files are present.")
                else:
                    f.write("Validation status: False\nMissing files: " + ", ".join(missing_files))
                    
            return validation_status
        
        except Exception as e:
            raise e