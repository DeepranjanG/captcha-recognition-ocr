import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from ocr.constants import DEVICE
from ocr.exception import CustomException
from ocr.components.data_preparation import DataPreparation


try:
    if __name__ == '__main__':

        # best_model_path = r"D:\Project\DL\custom-ocr-pytorch\artifacts\PredictModel\model.pt"

        best_model_path = r"D:\Project\DL\custom-ocr-pytorch\artifacts\11_28_2022_18_30_48\ModelTrainerArtifacts\model.pt"

        valid_dir = r"D:\Project\DL\custom-ocr-pytorch\artifacts\11_28_2022_23_11_21\DataIngestionArtifacts\val"

        valid_dataset = DataPreparation(valid_dir)

        valid_loader = DataLoader(valid_dataset, batch_size=8,
                                  num_workers=2, shuffle=False)

        ocr = torch.load(best_model_path, map_location=torch.device(DEVICE))

        valid_losses = []
        tot_val_loss = 0

        for i, (images, labels) in enumerate(valid_loader):
            logits, val_loss = ocr.val_step(images, labels)

            tot_val_loss += val_loss.item()

            val_loss = tot_val_loss / len(valid_loader.dataset)

            valid_losses.append(val_loss)

        print(valid_losses)

        print(np.mean(valid_losses))


except Exception as e:
    raise CustomException(e, sys) from e

