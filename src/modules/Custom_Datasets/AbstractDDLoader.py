import multiprocessing
import torch

from datadings.reader import MsgpackReader
from abc import abstractmethod


class AbstractDDLoader(torch.utils.data.Dataset):
    def __init__(self):

        self.lock = multiprocessing.Lock()

        # self.reader = MsgpackReader(self.datafile)
        self.__len = len(MsgpackReader(self.datafile, buffering=4 * 1024))
        self.__reader = None

    @property
    def reader(self):
        if self.__reader is None:
            self.__reader = MsgpackReader(self.datafile, buffering=4 * 1024)
        return self.__reader

    @abstractmethod
    def __getitem__(self, index):
        pass

    def __len__(self):
        return self.__len

    def get_num_classes(self):
        return self.num_classes

    def get_class_names(self):
        return self.class_names