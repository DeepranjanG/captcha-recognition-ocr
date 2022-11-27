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


@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR,
                                                                   DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRAIN_TRANSFORM_DATA_ARTIFACT_DIR = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,
                                                              DATA_TRANSFORMATION_TRAIN_DIR)
        self.VALID_TRANSFORM_DATA_ARTIFACT_DIR = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,
                                                              DATA_TRANSFORMATION_VALID_DIR)
        self.TRAIN_TRANSFORM_OBJECT_FILE_PATH = os.path.join(self.TRAIN_TRANSFORM_DATA_ARTIFACT_DIR,
                                                             DATA_TRANSFORMATION_TRAIN_FILE_NAME)
        self.VALID_TRANSFORM_OBJECT_FILE_PATH = os.path.join(self.VALID_TRANSFORM_DATA_ARTIFACT_DIR,
                                                             DATA_TRANSFORMATION_VALID_FILE_NAME)


@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.MODEL_TRAINER_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR,
                                                             MODEL_TRAINER_ARTIFACTS_DIR)
        self.TRAINED_MODEL_PATH = os.path.join(self.MODEL_TRAINER_ARTIFACTS_DIR, MODEL_NAME)
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_WORKERS = NUM_WORKERS
        self.WEIGHT_DECAY = WEIGHT_DECAY
        self.MOMENTUM = MOMENTUM
        self.EPOCHS = EPOCHS
        self.LR = LR


@dataclass
class ModelPusherConfig:

    def __init__(self):
        self.TRAINED_MODEL_PATH = os.path.join(os.getcwd(),ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR)
        self.BUCKET_NAME = BUCKET_NAME
        self.MODEL_NAME = MODEL_NAME



