import os
from torch.utils.data import Dataset
from Tools.utils import pil_loader
import numpy as np
import random

class StateFarm(Dataset):
    """
        The state farm dataset, 
    """
    def __init__(self, ins_file, transform=None):
        assert os.path.isfile(ins_file)
        self.transform = transform
        self.datalist = self.read_ins(ins_file)
        self.length = len(self.datalist)
        self.associ_dict = self.get_subject_dict(self.datalist)
        self.driver_ids = list(self.associ_dict.keys())

    def __getitem__(self, index: int):
        img_path_0, label_0, id_0 = self.datalist[index].split(',')
        img_0 = pil_loader(img_path_0)
        label_0 = int(label_0)
        
        sia_label = index % 2
        if sia_label == 0:  # even index -> same person
            img_path_1, label_1 = random.choice(self.associ_dict[id_0])
        else:               # odd index -> different person
            driver_id_ = self.driver_ids.copy()
            driver_id_.remove(id_0)
            id_1 = random.choice(driver_id_)
            img_path_1, label_1 = random.choice(self.associ_dict[id_1])

        img_1 = pil_loader(img_path_1)
        if self.transform:
            img_0 = self.transform(img_0)
            img_1 = self.transform(img_1)

        return img_0, img_1, label_0, sia_label 

    @staticmethod
    def read_ins(ins_file):
        with open(ins_file, 'r') as f:
            lines = f.readlines()
            lines = [l.rstrip() for l in lines]
        return lines
    
    @staticmethod
    def get_subject_dict(lines):
        subject_dict = {}
        for l in lines:
            img_path, label, id = l.split(',')
            try:
                subject_dict[id].append((img_path, label))
            except KeyError:
                subject_dict[id] = [(img_path, label)]
        return subject_dict

    def __len__(self):
        return self.length

        
class StateFarm_Test(StateFarm):
    def __init__(self, ins_file, transform):
        super(StateFarm_Test, self).__init__(ins_file, transform=transform)

    def __getitem__(self, index: int):
        img_path = self.datalist[index]
        img_0 = pil_loader(img_path)
        if self.transform:
            img_0  = self.transform(img_0)
        return img_0
    