import os
import sys
from torch.utils.data import DataLoader
from ocr.logger import logging
from ocr.utils import load_object
from ocr.exception import CustomException
from ocr.entity.config_entity import ModelTrainerConfig
from ocr.entity.artifact_entity import DataTransformationArtifacts, ModelTrainerArtifacts


class ModelTrainer:

    def __init__(self, model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifacts:DataTransformationArtifacts):
        """
        :param data_transformation_artifacts: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model trainer
        """

        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifacts = data_transformation_artifacts

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:

        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            train_dataset = load_object(self.data_transformation_artifacts.transformed_train_object)
            valid_dataset = load_object(self.data_transformation_artifacts.transformed_valid_object)

            logging.info("Saved prepared datasets")

            train_loader = DataLoader(train_dataset, batch_size=self.model_trainer_config.BATCH_SIZE,
                                      num_workers=self.model_trainer_config.NUM_WORKERS, shuffle=True)

            valid_loader = DataLoader(valid_dataset, batch_size=self.model_trainer_config.BATCH_SIZE,
                                      num_workers=self.model_trainer_config.NUM_WORKERS, shuffle=False)



        except Exception as e:
            raise CustomException(e, sys) from e