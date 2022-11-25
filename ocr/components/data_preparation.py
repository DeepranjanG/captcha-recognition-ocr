import os
import sys
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from ocr.exception import CustomException
from ocr.logger import logging


class DataPreparation(Dataset):

    logging.info(f"Data preparation pipeline started")

    def __init__(self, file_path):
        path_list = os.listdir(file_path)
        abspath = os.path.abspath(file_path)

        self.img_list = [os.path.join(abspath, path) for path in path_list]

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        try:

            return len(self.img_list)

        except Exception as e:
            raise CustomException(e, sys) from e

    def __getitem__(self, item):
        try:
            path = self.img_list[item]

            label = os.path.basename(path).split('.')[0].lower().split()
            img = Image.open(path).convert('RGB')

            img_tensor = self.transform(img)

            logging.info(f"Data preparation pipeline completed")

            return img_tensor, label
        except Exception as e:
            raise CustomException(e, sys) from e

