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
TRAINED_MODEL_DIR = 'TrainedModel'
TRAINED_MODEL_NAME = 'model.pt'
TRAINED_BATCH_SIZE = 2
TRAINED_SHUFFLE = False
TRAINED_NUM_WORKERS = 4
EPOCH = 1

# Model evaluation constants
MODEL_EVALUATION_ARTIFACTS_DIR = 'ModelEvaluationArtifacts'
MODEL_EVALUATION_FILE_NAME = 'loss.csv'



