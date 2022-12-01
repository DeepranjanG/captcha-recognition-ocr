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


# jpg_to_png(train_path, train_path_png, train_files)
# jpg_to_png(valid_path, valid_path_png, valid_files)



# import required libraries
import torch
import torchvision.transforms as T
from PIL import Image

# read the input image
img = Image.open(r'D:\Project\DL\custom-ocr-pytorch\data\train\261.jpg')

# define a transform to rotate he input image
# transform = T.RandomRotation(degrees=(60,90), expand=True)

# define a transform to horizontally flip an image
# randomly with a given probability
transform = T.RandomVerticalFlip(p=1)

# rotate the input image using above defined trasnform
img = transform(img)

# dispaly the rotated image
img.show()


def image_rotate(path):

    angle = 90
    input_image = Image.open(path)
    output = input_image.rotate(angle, Image.NEAREST, expand = True, fillcolor = (255,255,255))
    output.save("data/1.jpg")


# image_rotate(r'D:\Project\DL\custom-ocr-pytorch\data\train\000.jpg')