from copy import deepcopy
import os
import gc
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from geom_median.torch import compute_geometric_median
from Models.autoencoders import CVAE
from Models.MLP import MLP
from Utils.Utils import Utils
import datetime 
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime

from Server.Server import Server

class DefenseServer(Server):
    def __init__(self, cf=None, model=None, attack_type=None, attacker_ratio=None, dataset = None, experiment_name = 'debug'):
        super().__init__(cf, model, attack_type, attacker_ratio, dataset)

        # Saving directory
        self.dir_path = f"Results/{experiment_name}/{self.dataset}/FedMC/{self.cf['data_dist']}_{int(attacker_ratio * 100)}_{attack_type}"
        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)
        os.makedirs(self.dir_path)

        # Defense parameters
        self.activation_size = cf["cvae_input_dim"]
        self.activation_samples = cf["FEMNIST_num_clients_trigger"] if self.dataset == "FEMNIST" else cf["size_trigger"]
        self.num_classes = cf["num_classes"]
        self.config_cvae = {"cvae_nb_ep": cf["cvae_nb_ep"],
                            "cvae_lr": cf["cvae_lr"],
                            "cvae_wd": cf["cvae_wd"],
                            "cvae_gamma": cf["cvae_gamma"],}
        
        # Initiate CVAE
        self.cvae_trained = False
        self.cvae = CVAE(input_dim=cf["cvae_input_dim"],
            condition_dim=self.cf["condition_dim"],
            hidden_dim=self.cf["hidden_dim"],
            latent_dim=self.cf["latent_dim"]).to(self.device)
        
    def prepare_defense(self):
        pass 

    def compute_client_errors(self, clients): 
        pass 

    def filter_clients(self, clients): 
        # No defense
        return clients