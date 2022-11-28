import sys
import torch
import numpy as np
from PIL import Image
from ocr.constants import DEVICE
from torchvision import transforms
from ocr.exception import CustomException

try:

    best_model_path = r"D:\Project\DL\custom-ocr-pytorch\artifacts\11_28_2022_18_30_48\ModelTrainerArtifacts\model.pt"

    ocr = torch.load(best_model_path, map_location=torch.device(DEVICE))

    path = r"D:\Project\DL\custom-ocr-pytorch\artifacts\11_28_2022_18_05_16\DataIngestionArtifacts\train\2b827.png"

    image = Image.open(path).convert('RGB')

    convert_tensor = transforms.ToTensor()
    tensor_image = convert_tensor(image)

    logits = ocr.predict(tensor_image.unsqueeze(0))
    pred_text = ocr.decode(logits.cpu())

    print(pred_text)

except Exception as e:
    raise CustomException(e, sys) from e