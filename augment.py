import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import torchvision.transforms.functional as TF
import random
import matplotlib.pyplot as plt
import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch


def load_labels(path):
    df = pd.read_csv(path)
    tags = ['haze', 'primary', 'agriculture', 'clear', 'water', 'habitation', 'road', 'cultivation', 'slash_burn',
         'cloudy', 'partly_cloudy', 'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming',
         'selective_logging', 'blow_down']
    split_tags = [row.split(" ") for row in df["tags"]]
    mlb = MultiLabelBinarizer(classes=tags)
    mlb.fit(split_tags)
    img_labels = mlb.transform(split_tags).astype('float32')
    
    return img_labels, df


def load_image(path, idx, df):
    img_path = os.path.join(path, df.iloc[idx, 0] + '.jpg')
    image = read_image(img_path)

    return image


def transform_image(image):
    angle = random.randint(-30, 30)
    image = TF.rotate(image, angle)

    return image


def show_image(image):
    plt.imshow(  image.permute(1, 2, 0)  )


def main():
    csv_path = "/home/gro/Documents/Uni/Erasmus/Vorlesungen/DLVR/Project/datasets/deeplearningproject/train_classes.csv"
    img_base_path = "/home/gro/Documents/Uni/Erasmus/Vorlesungen/DLVR/Project/datasets/deeplearningproject/train-jpg/"

    img_labels, df = load_labels(csv_path)
    img = load_image(img_base_path, 0, df)
    trans_img = transform_image(img)

    show_image(trans_img)



if __name__ == "__main__":
    main()
