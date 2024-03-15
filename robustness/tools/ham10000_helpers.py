import os
import csv
import numpy as np
import torch as ch
import torch.utils.data as data
import robustness.data_augmentation as da
from robustness import imagenet_models
from PIL import Image

target_transform_ham = ch.Tensor
ALL_CLASSES_HAM10000 = ['bkl' 'nv' 'df' 'mel' 'vasc' 'bcc' 'akiec']

class HAM10000Dataset(data.Dataset):
  def __init__(self, root, train=True, train_labels_df=None, val_labels_df=None, transform=None, target_transform=None, **kwargs):
    self.img_dir = root
    self.transform = transform
    self.target_transform = target_transform

    image_label_dict = {}
    class_map = { code: i for i, code in enumerate(ALL_CLASSES_HAM10000) }
    class_counts = { i: 0 for i in range(7) }

    labels_df = train_labels_df if train else val_labels_df
    num_data = labels_df.shape[0]

    for image in os.listdir(self.img_dir):
      id = image.split(".")[0]
      if (id not in labels_df["image_id"].values): continue

      labels_row = labels_df.loc[labels_df["image_id"] == id].iloc[0]
      label = class_map[labels_row["dx"]]

      image_label_dict[id] = label
      class_counts[label] += 1

    # self.items is a list of tuples like: [ ('ISIC_0034317', 1), ('ISIC_0034315', 5), ... ]
    self.items = list(image_label_dict.items())
    self.class_counts = class_counts
    print(f"Class counts:", class_counts)

  def __len__(self):
    return len(self.items)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, f"{self.items[idx][0]}.jpg")
    image = Image.open(img_path)

    label = self.items[idx][1]

    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)

    return image, label