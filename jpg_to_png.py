from PIL import Image
import os

dataset = os.path.join(os.getcwd(), "data")
train_path = os.path.join(dataset, "train")
valid_path = os.path.join(dataset, "valid")

train_path_png = os.path.join(dataset, "train_png")
valid_path_png = os.path.join(dataset, "valid_png")


train_files = os.listdir(train_path)
valid_files = os.listdir(valid_path)


def jpg_to_png(current_location, destination, filenames):
    for i in range(len(filenames)):
        im1 = Image.open(os.path.join(current_location, filenames[i]))
        im1.save(os.path.join(destination, filenames[i].replace(".jpg", ".png")))

    return "Conversion done"


jpg_to_png(train_path, train_path_png, train_files)
jpg_to_png(valid_path, valid_path_png, valid_files)

