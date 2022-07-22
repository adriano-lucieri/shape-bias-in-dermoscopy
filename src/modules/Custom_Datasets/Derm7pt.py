import torch
import io
from PIL import Image
import numpy as np
from datadings.reader import MsgpackReader
import multiprocessing
import os

class Derm7pt(torch.utils.data.Dataset):
    def __init__(self, split='test', target_label='diagnosis', basepath=None, transforms=None, seed=42):

        np.random.seed(seed)

        self.transforms = transforms
        self.basepath = basepath
        self.target_label = target_label

        self.class_names = {
            'diagnosis': ['BCC', 'NV', 'MEL', 'SK', 'MISC'],
            'diagnosis_NV-MEL': ['NV', 'MEL'],
            'pigment_network': ['absent', 'typical', 'atypical'],
            'streaks': ['absent', 'regular', 'irregular'],
            'pigmentation': ['absent', 'diffuse regular', 'localized regular', 'diffuse irregular',
                             'localized irregular'],
            'regression_structures': ['absent', 'blue areas', 'white areas', 'combinations'],
            'dots_and_globules': ['absent', 'regular', 'irregular'],
            'blue_whitish_veil': ['absent', 'present'],
            'vascular_structures': ['absent', 'present'],
            }
        self.class_names = self.class_names[self.target_label]
        self.num_classes = len(self.class_names)

        self.lock = multiprocessing.Lock()

        if split == 'train':
            self.datafile = os.path.join(self.basepath, f'Derm7pt/datadings/Derm7pt_strat_perTarget/Derm7pt_strat_{self.target_label}/train.msgpack')
        elif split == 'val':
            self.datafile = os.path.join(self.basepath, f'Derm7pt/datadings/Derm7pt_strat_perTarget/Derm7pt_strat_{self.target_label}/val.msgpack')
        elif split == 'test':
            self.datafile = os.path.join(self.basepath, f'Derm7pt/datadings/Derm7pt_strat_perTarget/Derm7pt_strat_{self.target_label}/test.msgpack')

        if self.target_label == 'diagnosis_NV-MEL':
            self.target_label = 'diagnosis'
        
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
        label = sample[self.target_label]

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

            self.labels.append(sample[self.target_label])

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
