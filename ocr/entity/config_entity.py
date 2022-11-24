from dataclasses import dataclass
from ocr.constants import *
import os


@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.BUCKET_NAME: str = BUCKET_NAME
        self.ZIP_FILE_NAME:str = ZIP_FILE_NAME
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.TRAIN_DATA_ARTIFACT_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_TRAIN_DIR)
        self.VALID_DATA_ARTIFACT_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_VALID_DIR)
        self.ZIP_FILE_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, self.ZIP_FILE_NAME)