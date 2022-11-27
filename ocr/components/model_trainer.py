import os
import sys
import torch
import torch.optim as optim
from ocr.logger import logging
from ocr.constants import DEVICE
from ocr.ml.custom_model import Model
from ocr.utils import load_object
from torch.utils.data import DataLoader
from ocr.exception import CustomException
from ocr.entity.config_entity import ModelTrainerConfig
from ocr.entity.artifact_entity import DataTransformationArtifacts, ModelTrainerArtifacts


class ModelTrainer:

    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifacts: DataTransformationArtifacts):
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

            logging.info("Loaded dataset into the DataLoader and can iterate through the dataset")

            model = Model()

            logging.info("Loaded custom model")

            optimizer = optim.SGD(model.crnn.parameters(), lr=self.model_trainer_config.LR,
                                  nesterov=True, weight_decay=self.model_trainer_config.WEIGHT_DECAY,
                                  momentum=self.model_trainer_config.MOMENTUM)

            train_losses, valid_losses = model.train(self.model_trainer_config.EPOCHS,
                                                     optimizer, train_loader, valid_loader, DEVICE)

            logging.info(f"Saved the train_losses: {train_losses} and valid losses: {valid_losses}")

            os.makedirs(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, exist_ok=True)
            torch.save(model, self.model_trainer_config.TRAINED_MODEL_PATH)

            logging.info(f"Saved the trained model")

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifacts}")

            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
