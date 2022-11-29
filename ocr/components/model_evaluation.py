import os
import sys
import numpy as np
import torch
from ocr.constants import DEVICE
from ocr.logger import logging
from ocr.utils import load_object
from torch.utils.data import DataLoader
from ocr.exception import CustomException
from ocr.configuration.gcloud_syncer import GCloudSync
from ocr.entity.config_entity import ModelEvaluationConfig
from ocr.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts, DataTransformationArtifacts


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifacts,
                 data_transformation_artifacts: DataTransformationArtifacts):
        """
        :param model_evaluation_config: Configuration for model evaluation
        :param data_transformation_artifacts: Output reference of data transformation artifact stage
        :param model_trainer_artifacts: Output reference of model trainer artifact stage
        """

        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        self.gcloud = GCloudSync()

    def get_best_model_from_gcloud(self) -> str:
        """
        :return: Fetch best model from gcloud storage and store inside best model directory path
        """
        try:
            logging.info("Entered the get_best_model_from_gcloud method of Model Evaluation class")

            os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH, exist_ok=True)

            self.gcloud.sync_folder_from_gcloud(self.model_evaluation_config.BUCKET_NAME,
                                                self.model_evaluation_config.MODEL_NAME,
                                                self.model_evaluation_config.BEST_MODEL_DIR_PATH)

            best_model_path = os.path.join(self.model_evaluation_config.BEST_MODEL_DIR_PATH,
                                           self.model_evaluation_config.MODEL_NAME)
            logging.info("Exited the get_best_model_from_gcloud method of Model Evaluation class")
            return best_model_path
        except Exception as e:
            raise CustomException(e, sys) from e

    def evaluate(self, model, data_loader):
        """

        :param model: Currently trained model or best model from gcloud storage
        :param data_loader: Data loader for validation dataset
        :return: loss
        """
        try:
            logging.info("Loading model and valid data loader for evaluation")
            losses = []
            tot_val_loss = 0
            for i, (images, labels) in enumerate(data_loader):
                logits, loss = model.val_step(images, labels)
                tot_val_loss += loss.item()
                loss = tot_val_loss / len(data_loader.dataset)
                losses.append(loss)
            logging.info(f"Return average loss on the test or valid dataset: {np.mean(losses)}")
            return np.mean(losses)

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
            Method Name :   initiate_model_evaluation
            Description :   This function is used to initiate all steps of the model evaluation

            Output      :   Returns model evaluation artifact
            On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Initiate Model Evaluation")
        try:
            logging.info("Loading validation data for model evaluation")
            valid_dataset = load_object(self.data_transformation_artifacts.transformed_valid_object)
            valid_loader = DataLoader(valid_dataset, batch_size=self.model_evaluation_config.BATCH_SIZE,
                                      num_workers=self.model_evaluation_config.NUM_WORKERS, shuffle=False)

            logging.info("Loading currently trained model")
            trained_model = torch.load(self.model_trainer_artifacts.trained_model_path, map_location=torch.device(DEVICE))
            trained_model_loss = self.evaluate(trained_model, valid_loader)

            logging.info("Fetch best model from gcloud storage")
            best_model_path = self.get_best_model_from_gcloud()

            logging.info("Check is best model present in the gcloud storage or not ?")
            if os.path.isfile(best_model_path) is False:
                is_model_accepted = True
                logging.info("glcoud storage model is false and currently trained model accepted is true")

            else:
                logging.info("Load best model fetched from gcloud storage")
                best_model = torch.load(best_model_path, map_location=torch.device(DEVICE))
                best_model_loss = self.evaluate(best_model, valid_loader)

                logging.info("Comparing loss between best_model_loss and trained_model_loss ? ")
                if best_model_loss > trained_model_loss:
                    is_model_accepted = True
                    logging.info("Trained model not accepted")
                else:
                    is_model_accepted = False
                    logging.info("Trained model accepted")

            model_evaluation_artifacts = ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
            return model_evaluation_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e


