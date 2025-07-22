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
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
from torch.nn.functional import cosine_similarity

from Server.Server import Server

class DefenseServer(Server):
    def __init__(self, cf=None, model=None, attack_type=None, attacker_ratio=None, dataset = None, sampler = None, experiment_name = 'debug'):
        super().__init__(cf, model, attack_type, attacker_ratio, dataset, sampler)

        # Saving directory
        self.dir_path = f"Results/{experiment_name}/{self.dataset}/{sampler.name}/FLEDGE/{self.cf['data_dist']}_{int(attacker_ratio * 100)}_{attack_type}"
        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)
        os.makedirs(self.dir_path)
        
    def prepare_defense(self):
        pass 

    def compute_client_errors(self, clients): 
        local_models = [client.model for client in clients]
        return self.get_scores(self.global_model, local_models)

    def filter_clients(self, clients): 
        scores = self.compute_client_errors(clients)
        xs, ys = self.get_kde(scores)
        groups = self.get_data_groups(xs, ys, scores)
        benign_key = str(min([int(k) for k in groups.keys()])) # closest to 0
        benign_indexes = groups[benign_key]
        print(groups)
        return [clients[index] for index in benign_indexes]
    
    def get_scores(self, global_model, local_models):
        scores = []
        for m in local_models:
            distance = self.cosine_between_models(global_model, m).detach().item()
            scores.append(1-distance)
        return scores
    
    def get_data_groups(self, x, y, scores):
        mins = list(argrelextrema(y, np.less)[0])
        mins.append(len(x))
        initial = 0
        groups = {}
        for i, m in enumerate(mins):
            r = x[initial:m]
            indexes = []
            for j, s in enumerate(scores):
                if s >= min(r) and s <= max(r):
                    indexes.append(j)
            groups[str(i)] = indexes
            initial = m
        return groups
    
    def get_kde(self, scores):
        #kde = gaussian_kde(np.array(scores))
        kde = GaussianKde(np.array(scores))
        xs = np.linspace(min(scores)-np.std(scores), max(scores)+np.std(scores), 2000)
        kde.covariance_factor = lambda : .5
        kde._compute_covariance()
        ys = kde(xs)
        return xs, ys
    
    def cosine_between_models(self, m1, m2):
        v1 = self.get_one_vec_params(m1)
        v2 = self.get_one_vec_params(m2)
        return cosine_similarity(v1, v2, dim=0).cpu()
    
    def get_one_vec_params(self, model):
        """
        Converts a model, given as dictionary type, to a single vector
        """
        weight_list = []
        # Iterate over all parameters in the model
        for param in model.parameters():
            # Flatten the parameter tensor and append it to the list
            weight_list.append(param.view(-1))
        # Concatenate all parameter tensors into a single vector
        weight_vector = torch.cat(weight_list)
        return weight_vector

from numpy import (atleast_2d,cov,pi)
from scipy import linalg, special

class GaussianKde(gaussian_kde):
    """
    Drop-in replacement for gaussian_kde that adds the class attribute EPSILON
    to the covmat eigenvalues, to prevent exceptions due to numerical error.
    """

    EPSILON = 1e-20 # adjust this at will

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and Cholesky decomp of covariance
        if not hasattr(self, '_data_cho_cov'):
            self._data_covariance = atleast_2d(cov(self.dataset, rowvar=1,
                                               bias=False,
                                               aweights=self.weights))
            
            #self._data_covariance[np.isnan(self._data_covariance)] = 0
            #self._data_covariance[np.isinf(self._data_covariance)] = 1e5
            
            self._data_covariance += self.EPSILON * np.eye(len(self._data_covariance))
            
            self._data_cho_cov = linalg.cholesky(self._data_covariance,
                                                 lower=True)

        self.covariance = self._data_covariance * self.factor**2
        self.cho_cov = (self._data_cho_cov * self.factor).astype(np.float64)
        self.log_det = 2*np.log(np.diag(self.cho_cov
                                        * np.sqrt(2*pi))).sum()