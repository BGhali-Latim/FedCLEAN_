import os
import json
import numpy as np
import random
import torch
from math import floor
from statistics import mean

from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets
from torchvision.transforms import transforms
import torch.nn.functional as F
import gc
import h5py

import torchvision.transforms.functional as TF

from Client.Client import Client
from custom_datasets.Datasets import SyntheticLabeledDataset, MixupDataset
from custom_datasets import FEMNIST_pytorch 

from Utils import Utils

class ClientSampler(): 
    def __init__(self, config) -> None:
        self.num_clients = config["num_clients"]
        self.batch_size = config["batch_size"]
        self.name = "IID"
        
        # Data augmentation 
        #self.mixup = config["Mixup"]
    
    def compute_dataset_mean_std(self,dataset):
            # Create a DataLoader to iterate through the dataset
            dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=2)
            # Initialize variables to hold the sum and sum of squares of pixel values
            mean = torch.zeros(3)
            sqr_mean = torch.zeros(3)
            nb_samples = 0
                        
            for data in dataloader:
                imgs, _ = data
                batch_samples = imgs.size(0)
                imgs = imgs.view(batch_samples, imgs.size(1), -1)
                mean += imgs.mean(2).sum(0)
                sqr_mean += torch.square(imgs).mean(2).sum(0)
                nb_samples += batch_samples
                        
            mean /= nb_samples
            sqr_mean /= nb_samples
            var = sqr_mean - torch.square(mean)
            std = torch.sqrt(var)
                        
            print(f'Mean: {mean}')
            print(f'Std: {std}')
            return None

    def load_dataset(self, dataset): 
        if dataset == "MNIST" :
            trans_mnist_train = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize((0.1307,), (0.3081,))
                    ])
            train_data = datasets.MNIST(root='./data', 
                                             train=True, 
                                             transform=trans_mnist_train, 
                                             download=True)

        elif dataset == "FashionMNIST" :
            trans_fashionmnist_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,), (0.3530,))
                    ])
            train_data = datasets.FashionMNIST(root='./data', 
                                                    train=True, 
                                                    transform=trans_fashionmnist_train, 
                                                    download=True)

        elif dataset == "CIFAR10" : 
            trans_cifar_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(([0.4246, 0.4148, 0.3838]), ([0.2829, 0.2779, 0.2845])),
            #transforms.Normalize(([0.5,0.5,0.5]), ([0.5,0.5,0.5])),
            ])
            train_data = datasets.CIFAR10(root='./data', 
                                               train=True, 
                                               transform=trans_cifar_train, 
                                               download=True)
            #train_data.targets = torch.tensor(train_data.targets)
            #train_data.data = torch.tensor(train_data.data)
            
        elif dataset == "FEMNIST": 
            train_data = Utils.load_from_hdf5(source = './data/write_digits.hdf5', train = True)
        
        return train_data
    
    def get_test_data(self, size_trigger, dataset):
        if dataset == "MNIST" :
            trans_mnist_train = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize((0.1307,), (0.3081,))
                    ])
            data_test = datasets.MNIST(root='./data',
                                        train=False,
                                        transform=trans_mnist_train,
                                        download=True)
            
        elif dataset == "FashionMNIST" :
            trans_fashionmnist_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,), (0.3530,))
                    ])
            data_test = datasets.FashionMNIST(root='./data',
                                        train=False,
                                        transform=trans_fashionmnist_test,
                                        download=True)
            
        elif dataset == "CIFAR10" :
            trans_cifar_test = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                                                #transforms.Normalize(([0.5,0.5,0.5]), ([0.5,0.5,0.5])),
                                                ])
            data_test = datasets.CIFAR10(root='./data',
                                        train=False,
                                        transform=trans_cifar_test,
                                        download=True)
            
        elif dataset == "FEMNIST" : 
            data_test = Utils.load_from_hdf5(source = './data/write_digits.hdf5', 
                                        train=False)
            
        size_test = len(data_test) - size_trigger
        trigger_set, validation_set = random_split(data_test, [size_trigger, size_test])
        # Create data loaders
        trigger_loader = DataLoader(trigger_set, batch_size=size_trigger, shuffle=False, drop_last=False) if size_trigger else None
        test_loader = DataLoader(validation_set, batch_size=size_test, shuffle=False, drop_last=False)
        return trigger_loader, test_loader
    
    def distribute_iid_data(self, dataset): 
            train_data = self.load_dataset(dataset)
            data_size = len(train_data) // self.num_clients
            return [DataLoader(Subset(train_data, range(i * data_size, (i + 1) * data_size)), batch_size=self.batch_size, shuffle=True, drop_last=False)
                    for i in range(self.num_clients)]
    
    def distribute_non_iid_data(self, dataset): 
        pass

    def perform_augmentation(self, client_dataset): 
        if self.mixup : 
            client_dataset = MixupDataset(client_dataset, alpha = 1)
        return client_dataset