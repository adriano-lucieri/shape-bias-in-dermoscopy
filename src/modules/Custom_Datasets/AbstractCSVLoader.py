import torch
import csv
import cv2
import os

import numpy as np


class AbstractCSVLoader(torch.utils.data.Dataset):
    def __init__(self, transforms=None, seed=42):

        np.random.seed(seed)

        self.transforms = transforms
        self.num_classes = 0
        self.class_names = []
        self.dataset_path = None
        self.datafile = None
        self.num_channels = 3 # Default

    def __getitem__(self, index):
        image = cv2.imread(self.filepaths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']

        label = self.labels[index]

        return image, label, self.filepaths[index]

    def __len__(self):
        return len(self.labels)

    def _get_filenames_and_labels(self):

        with open(self.datafile, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter='|')

            self.filepaths = []
            self.labels = []

            for row in reader:
                self.filepaths.append(os.path.join(self.dataset_path, row[0]))
                self.labels.append(int(row[1]))

        # Shuffle
        p = np.random.permutation(np.arange(len(self.filepaths)))
        self.filepaths = np.array(self.filepaths)[p]
        self.labels = np.array(self.labels)[p]

    def get_num_classes(self):
        return self.num_classes

    def get_num_channels(self):
        return self.num_channels

    def get_lst_class_names(self):
        return self.class_names

    def get_class_weights(self):
        values, counts = np.unique(self.labels, return_counts=True)

        return np.sum(counts) / counts

    def get_sample_weights(self):

        class_weights = self.get_class_weights()

        sample_weights = [0] * len(self.labels)
        for idx, val in enumerate(self.labels):
            sample_weights[idx] = class_weights[val]
        return sample_weights