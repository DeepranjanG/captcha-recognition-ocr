from dataclasses import dataclass


# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    train_file_path: str
    valid_file_path: str


@dataclass
class DataTransformationArtifacts:
    transformed_train_object: str
    transformed_valid_object: str


@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str


@dataclass
class ModelPusherArtifacts:
    bucket_name: str
