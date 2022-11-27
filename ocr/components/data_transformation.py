import os
import sys
from ocr.logger import logging
from ocr.exception import CustomException
from ocr.utils import save_object
from ocr.components.data_preparation import DataPreparation
from ocr.entity.config_entity import DataTransformationConfig
from ocr.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts


class DataTransformation:

    def __init__(self, data_ingestion_artifacts:DataIngestionArtifacts,
                 data_transformation_config: DataTransformationConfig):
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.data_transformation_config = data_transformation_config

    def initiate_data_transformation(self) -> DataTransformationArtifacts:

        try:

            logging.info("Entered the initiate_data_transformation method of Data transformation class")

            train_dataset = DataPreparation(self.data_ingestion_artifacts.train_file_path)

            logging.info(f"Training dataset prepared")

            valid_dataset = DataPreparation(self.data_ingestion_artifacts.valid_file_path)

            logging.info(f"Validation dataset prepared")

            save_object(self.data_transformation_config.TRAIN_TRANSFORM_OBJECT_FILE_PATH, train_dataset)
            save_object(self.data_transformation_config.VALID_TRANSFORM_OBJECT_FILE_PATH, valid_dataset)

            logging.info("Saved the transformed dataset object")

            data_transformation_artifacts = DataTransformationArtifacts(
                transformed_train_object=self.data_transformation_config.TRAIN_TRANSFORM_OBJECT_FILE_PATH,
                transformed_valid_object=self.data_transformation_config.VALID_TRANSFORM_OBJECT_FILE_PATH
            )

            logging.info("Exited the initiate_data_transformation method of Data transformation class")

            return data_transformation_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
