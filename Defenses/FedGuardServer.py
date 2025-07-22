from copy import deepcopy
import os
import gc
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from geom_median.torch import compute_geometric_median
from Models.autoencoders import GuardCVAE
from Models.MLP import MLP
from Utils.Utils import Utils
import datetime 
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset
from custom_datasets.Datasets import SyntheticLabeledDataset
import random

from Server.Server import Server

class DefenseServer(Server):
    def __init__(self, cf=None, model=None, attack_type=None, attacker_ratio=None, dataset = None, sampler = None, experiment_name = 'debug'):
        super().__init__(cf, model, attack_type, attacker_ratio, dataset, sampler)
        # Defense server
        self.defence = True

        # Saving directory
        self.dir_path = f"Results/{experiment_name}/{self.dataset}/{sampler.name}/FedGuard/{self.cf['data_dist']}_{int(attacker_ratio * 100)}_{attack_type}"
        print(self.dir_path)
        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)
        os.makedirs(self.dir_path)

        # Defense parameters
        self.cvae_training_config = {
            "nb_ep": cf["cvae_nb_ep"],
            "lr": cf["cvae_lr"],
            "wd": cf["cvae_wd"],
            "gamma": cf["cvae_gamma"],
        }
        
        # Initiate CVAE
        self.cvae_trained = False
        self.guardCvae = GuardCVAE(condition_dim=self.cf["condition_dim"]).to(self.device)
    
    # Defense core functions
    def prepare_defense(self):
        if not self.cvae_trained :
            self.prepare_local_cvae()

    def compute_client_errors(self, clients, latent_space_samples, condition_samples): 
        return self.compute_acc_loss(clients, latent_space_samples, condition_samples)
    
    def filter_clients(self, selected_clients): 
        # Error computations and handling step
        latent_space_samples = Utils.sample_from_normal(self.cf["nb_samples"],self.cf["latent_dim"], device=self.device)
        condition_samples = Utils.sample_from_cat(self.cf["nb_samples"], device=self.device)
        clients_acc_losses = self.compute_acc_loss(selected_clients, latent_space_samples, condition_samples)
        clients_acc_losses_np = np.array(clients_acc_losses)
        valid_values = clients_acc_losses_np[np.isfinite(clients_acc_losses_np)]
        max_of_acc_loss = np.max(valid_values)
        mean_of_acc_loss = np.mean(valid_values)
        clients_acc_loss_without_nan = np.where(np.isnan(clients_acc_losses_np) | (clients_acc_losses_np == np.inf), max_of_acc_loss,
                                          clients_acc_losses_np)
        # Detection and filtering step
        selected_clients_array = np.array(selected_clients)
        good_updates = selected_clients_array[clients_acc_loss_without_nan < mean_of_acc_loss]
        for client in selected_clients_array[clients_acc_loss_without_nan >= mean_of_acc_loss]:
            client.suspect()

        return good_updates
    
    # Utils used by defense
    def prepare_local_cvae(self):
        # Prepare local CVAE
        for idx, client in enumerate(self.clients) : 
            client.set_guardCvae(deepcopy(self.guardCvae)) 
            print(f"Training guardCvae for client {idx+1}/{len(self.clients)}")
            client.train_guardCvae(self.cvae_training_config) 
        self.cvae_trained = True
    
    def compute_acc_loss(self, selected_clients, latent_space_samples, condition_samples):
        client_acc_losses = []
        loss = torch.nn.MSELoss()
        
        for client in selected_clients :
            synthetic_ds = client.generate_synthetic_data(latent_space_samples, Utils.one_hot_encoding(condition_samples, self.cf["nb_classes"], self.device))
            eval_data = SyntheticLabeledDataset(synthetic_ds, condition_samples)
            eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=4)
            acc_loss = Utils.get_loss(client.model,self.device,eval_loader,loss).cpu()
            client_acc_losses.append(acc_loss)
        
        return client_acc_losses