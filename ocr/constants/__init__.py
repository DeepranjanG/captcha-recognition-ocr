import os
import torch
from datetime import datetime

# Common constants
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
BUCKET_NAME = 'custom-ocr-pytorch'
ZIP_FILE_NAME = 'dataset.zip'

use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")

APP_HOST = "0.0.0.0"
APP_PORT = 8080

# Data ingestion constants
DATA_INGESTION_ARTIFACTS_DIR = 'DataIngestionArtifacts'
DATA_INGESTION_TRAIN_DIR = 'train'
DATA_INGESTION_VALID_DIR = 'val'

# Data transformation constants
DATA_TRANSFORMATION_ARTIFACTS_DIR = 'DataTransformationArtifacts'
DATA_TRANSFORMATION_TRAIN_DIR = 'Train'
DATA_TRANSFORMATION_VALID_DIR = 'Valid'
DATA_TRANSFORMATION_TRAIN_FILE_NAME = "train.pkl"
DATA_TRANSFORMATION_VALID_FILE_NAME = "valid.pkl"


# Model Training Constants
MODEL_TRAINER_ARTIFACTS_DIR = 'ModelTrainerArtifacts'
MODEL_NAME = 'model.pt'
BATCH_SIZE = 8
NUM_WORKERS = 2
EPOCHS = 1
LR = 0.02
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.7
CHARS = 'abcdefghijklmnopqrstuvwxyz0123456789 '
# CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ '
VOCAB_SIZE = len(CHARS) + 1


# Model evaluation constants
MODEL_EVALUATION_ARTIFACTS_DIR = 'ModelEvaluationArtifacts'
MODEL_EVALUATION_FILE_NAME = 'loss.csv'



