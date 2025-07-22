import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticLabeledDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        dt = self.data[idx, :]
        lbl = self.labels[idx]
        return dt, lbl
    
    #class OneClassDataset(Dataset): 
    #    def __init__(self, dataset, class_label):
    #        self.dataset = dataset

class MixupDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, alpha=1.0):
        self.dataset = dataset
        self.alpha = alpha

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image1, label1 = self.dataset[index]

        index2 = np.random.randint(0, len(self.dataset))
        image2, label2 = self.dataset[index2]

        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)

        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_label = lam * label1 + (1 - lam) * label2

        return mixed_image, mixed_label.type(torch.LongTensor)
