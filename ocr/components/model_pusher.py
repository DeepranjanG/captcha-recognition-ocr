
import sys
from ocr.logger import logging
from ocr.exception import CustomException
from ocr.configuration.gcloud_syncer import GCloudSync
from ocr.entity.config_entity import ModelPusherConfig


class ModelPusher:

    def __init__(self, model_pusher_config: ModelPusherConfig):
        """
        :param model_pusher_config: Configuration for model pusher
        """
        self.model_pusher_config = model_pusher_config
        self.gcloud = GCloudSync()

    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        """
            Method Name :   initiate_model_pusher
            Description :   This method initiates model pusher.

            Output      :    Model pusher artifact
        """
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")
        try:
            # Uploading the model to gcloud storage



            logging.info("Uploaded best model to gcloud storage")

            # Saving the model pusher artifacts
            model_pusher_artifact = ModelPusherArtifacts(
                bucket_name=self.model_pusher_config.BUCKET_NAME,
                s3_model_path=self.model_pusher_config.S3_MODEL_KEY_PATH,
            )
            logging.info("Exited the initiate_model_pusher method of ModelTrainer class")
            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys) from e

