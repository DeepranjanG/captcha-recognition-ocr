from dataclasses import dataclass


# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    train_file_path: str
    valid_file_path: str

    def to_dict(self):
        return self.__dict__


@dataclass
class DataTransformationArtifacts:
    transformed_train_object: str
    transformed_valid_object: str

    def to_dict(self):
        return self.__dict__


@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str

    def to_dict(self):
        return self.__dict__


@dataclass
class ModelPusherArtifacts:
    bucket_name: str

    def to_dict(self):
        return self.__dict__
