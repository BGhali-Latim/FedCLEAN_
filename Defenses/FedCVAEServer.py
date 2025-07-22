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

from Server.Server import Server

class DefenseServer(Server):
    def __init__(self, cf=None, model=None, attack_type=None, attacker_ratio=None, dataset = None, sampler = None, experiment_name = 'debug'):
        super().__init__(cf, model, attack_type, attacker_ratio, dataset, sampler)
        # Defense server
        self.defence = True

        # Saving directory
        self.dir_path = f"Results/{experiment_name}/{self.dataset}/{sampler.name}/FedCVAE/{self.cf['data_dist']}_{int(attacker_ratio * 100)}_{attack_type}"
        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)
        os.makedirs(self.dir_path)

        # CVAE initialisation
        self.cvae = CVAE(
            input_dim=self.cf["selected_weights_dim"],
            condition_dim=self.cf["condition_dim"],
            hidden_dim=self.cf["hidden_dim"],
            latent_dim=self.cf["latent_dim"]
        ).to(self.device)
    
    # Defense core functions
    def prepare_defense(self):
        # Track rounds for CVAE condition
        self.round = 0
        # Surrogate vector selection parameters
        total_weights = sum(param.numel() for param in self.global_model.parameters()  if param.dim() > 1)
        # selecting the indices that will be fed to the CVAE
        self.indices = np.random.choice(total_weights, self.cf["selected_weights_dim"], replace=False)

    def compute_client_errors(self, clients): 
        return self.compute_reconstruction_error(clients)
    
    def filter_clients(self, selected_clients): 
        # Error computations and handling step
        surrogate_weights = self.gen_surrogate_vectors(selected_clients)
        processed_vectors = self.process_surrogate_vectors(surrogate_weights)
        clients_re = self.compute_reconstruction_error(processed_vectors, self.round)
        clients_re_np = np.array(clients_re)
        mean_of_re = np.mean(clients_re_np)

        # Detection and filtering step
        selected_clients_np = np.array(selected_clients)
        good_updates = selected_clients_np[clients_re_np < mean_of_re]
        for client in selected_clients_np[clients_re_np >= mean_of_re]:
            client.suspect()
        
        # Increment rounds for next error computation 
        self.round +=1 

        return good_updates
    
    # Utils used by defense
    def one_hot_encoding(self, current_round):
        one_hot = torch.zeros(self.cf['condition_dim']).to(self.device)
        one_hot[current_round] = 1.0
        return one_hot
        
    def gen_surrogate_vectors(self, selected_clients):
        surrogate_vectors = [torch.cat([p.data.view(-1) for p in client.model.parameters() if p.dim() > 1])[
                self.indices].detach().cpu() for client in selected_clients]
        return surrogate_vectors
    
    def process_surrogate_vectors(self, surrogate_vectors):
        geo_median = compute_geometric_median(surrogate_vectors, weights=None, eps=self.cf["eps"], maxiter=self.cf["iter"])  # equivalent to `weights = torch.ones(n)`.
        geo_median = geo_median.median
        processed_vectors = [surrogate_vector - geo_median for surrogate_vector in surrogate_vectors]
        return processed_vectors
    
    def compute_reconstruction_error(self, processed_vectors, current_round):
        self.cvae.eval()
        clients_re = []
        condition = self.one_hot_encoding(current_round).unsqueeze(0).to(self.device)
        for processed_vector in processed_vectors:
            processed_vector = processed_vector.unsqueeze(0).to(self.device)
            recon_batch, _, _ = self.cvae(processed_vector, condition)
            mse = F.mse_loss(recon_batch, processed_vector, reduction='mean').item()
            clients_re.append(mse)
        return clients_re
    
    
