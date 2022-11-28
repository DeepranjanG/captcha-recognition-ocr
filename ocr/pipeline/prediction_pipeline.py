import os
import io
import sys
from PIL import Image
from ocr.logger import logging
from ocr.constants import *
from torchvision import transforms
from ocr.exception import CustomException
from ocr.configuration.gcloud_syncer import GCloudSync


class PredictionPipeline:
    def __init__(self):
        self.bucket_name = BUCKET_NAME
        self.model_name = MODEL_NAME
        self.model_path = os.path.join("artifacts", "PredictModel")
        self.gcloud = GCloudSync()

    def image_loader(self, image_bytes):
        """load image, returns cuda tensor"""
        logging.info("Entered the image_loader method of PredictionPipeline class")
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            convert_tensor = transforms.ToTensor()
            tensor_image = convert_tensor(image)
            logging.info("Exited the image_loader method of PredictionPipeline class")
            return tensor_image

        except Exception as e:
            raise CustomException(e, sys) from e

    def get_model_from_gcloud(self) -> str:
        """
        Method Name :   predict
        Description :   This method predicts the image.

        Output      :   Predictions
        """
        logging.info("Entered the get_model_from_gcloud method of PredictionPipeline class")
        try:
            # Loading the best model from s3 bucket
            os.makedirs(self.model_path, exist_ok=True)
            self.gcloud.sync_folder_from_gcloud(self.bucket_name, self.model_name, self.model_path)
            best_model_path = os.path.join(self.model_path, self.model_name)
            logging.info("Exited the get_model_from_gcloud method of PredictionPipeline class")
            return best_model_path

        except Exception as e:
            raise CustomException(e, sys) from e

    def prediction(self, best_model_path: str, image) -> float:
        logging.info("Entered the prediction method of PredictionPipeline class")
        try:
            model = torch.load(best_model_path, map_location=torch.device(DEVICE))
            logits = model.predict(image.unsqueeze(0))
            pred_text = model.decode(logits.cpu())
            return pred_text

        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self, data):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            image = self.image_loader(data)
            best_model_path: str = self.get_model_from_gcloud()
            predicted_text = self.prediction(best_model_path, image)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return predicted_text
        except Exception as e:
            raise CustomException(e, sys) from e

