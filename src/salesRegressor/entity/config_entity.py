from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    sales_file: Path
    store_file: Path
    cleaned_data_file: Path
    train_file: Path
    test_file: Path
    test_size: float

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_file: Path
    test_file: Path
    model_file: Path
    iterations: int
    learning_rate: float
    depth: int
    loss_function: str
    early_stopping_rounds: int
    verbose: int

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    test_data_path: Path
    metrics_file: Path