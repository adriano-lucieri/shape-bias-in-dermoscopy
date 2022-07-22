import torch
import io
from PIL import Image
import numpy as np
from datadings.reader import MsgpackReader
import multiprocessing
import os

class ISICb_NV_MEL_Augmentations(torch.utils.data.Dataset):
    def __init__(self, mode='test', basepath=None, augmentation_name=None, transforms=None, seed=42, portion=None):

        np.random.seed(seed)

        self.transforms = transforms
        self.basepath = basepath

        self.class_names = ['NV', 'MEL']
        self.num_classes = len(self.class_names)

        self.lock = multiprocessing.Lock()

        if mode == 'train':
            if portion is None:
                self.datafile = os.path.join(self.basepath, f'ISIC/datadings/Augmentations/{augmentation_name}/binary_NV-MEL/train.msgpack')
            else:
                target_file = os.path.join(self.basepath, f'ISIC/datadings/Augmentations/{augmentation_name}/binary_NV-MEL/train-{portion}.msgpack')
                print(f'Trying to load {target_file}')
                self.datafile = target_file
        elif mode == 'val':
            self.datafile = os.path.join(self.basepath, f'ISIC/datadings/Augmentations/{augmentation_name}/binary_NV-MEL/val.msgpack')
        elif mode == 'test':
            self.datafile = os.path.join(self.basepath, f'ISIC/datadings/Augmentations/{augmentation_name}/binary_NV-MEL/test.msgpack')

        self.__len = len(MsgpackReader(self.datafile, buffering=4 * 1024))
        self.__reader = None
        
    @property
    def reader(self):
        if self.__reader is None:
            self.__reader = MsgpackReader(self.datafile, buffering=4 * 1024)
        return self.__reader
    
    def __getitem__(self, index):
        reader = self.reader
        reader.seek(index)

        sample = reader.next()
        image = sample['image']
        label = int(sample['diagnosis'])

        image = Image.open(io.BytesIO(image))
        image = np.uint8(image)
        image = torch.from_numpy(image)
        image = image.permute((2, 0, 1))

        image = self.transforms(image)

        return image, label

    def __len__(self):
        return self.__len

    def get_num_classes(self):
        return self.num_classes

    def get_lst_class_names(self):
        return self.class_names

    def get_label_list(self):
        self.labels = []
        tmp_reader = MsgpackReader(self.datafile, buffering=4 * 1024)
        for i in range(len(tmp_reader)):
            tmp_reader.seek(i)
            sample = tmp_reader.next()

            self.labels.append(int(sample['diagnosis']))

    def get_class_weights(self):
        values, counts = np.unique(self.labels, return_counts=True)
    
        return np.sum(counts) / counts
    
    def get_sample_weights(self):
        self.get_label_list()
        class_weights = self.get_class_weights()
    
        sample_weights = [0] * len(self.labels)
        for idx, val in enumerate(self.labels):
            sample_weights[idx] = class_weights[val]
        return sample_weights
