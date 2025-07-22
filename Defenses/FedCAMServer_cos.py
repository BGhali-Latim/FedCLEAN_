from copy import deepcopy
import os
import gc
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from geom_median.torch import compute_geometric_median
from Models.autoencoders import ConvCVAE
from Models.MLP import MLP
from Utils.Utils import Utils
import datetime 
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset
import random

from Server.Server import Server


class DefenseServer(Server):
    def __init__(self, cf=None, model=None, attack_type=None, attacker_ratio=None, dataset = None, experiment_name = 'debug'):
        super().__init__(cf, model, attack_type, attacker_ratio, dataset)
        print("entered")

        # Saving directory
        self.dir_path = f"Results/experiment_name/{self.dataset}/FedCAM_cos/{self.cf['data_dist']}_{int(attacker_ratio * 100)}_{attack_type}"
        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)
        os.makedirs(self.dir_path)

        # Defense parameters
        self.activation_size = cf["cvae_input_dim"]
        self.activation_samples = cf["FEMNIST_num_clients_trigger"] if self.dataset == "FEMNIST" else cf["size_trigger"]
        self.num_classes = cf["num_classes"]
        
        # Initiate distances
        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.euclid_dist = torch.nn.PairwiseDistance(p=2, eps=1e-6)
        
        # Storage for geomed computation 
        # self.gm = None 
        # self.activations = None
    
    # Defense core functions
    def prepare_defense(self):
        pass

    def compute_client_errors(self, clients): 
        print("ok")
        # Storage 
        clients_re = []
        clients_act = torch.zeros(size=(len(clients), self.cf["size_trigger"], self.activation_size)).to(self.device)

        # Get activations
        with torch.no_grad() :
            for client_nb, client_model in enumerate(clients):
                for data, label in self.trigger_loader:
                    data, label = data.to(self.device), label.to(self.device)
                    activation = client_model.model.get_activations(data)
                    clients_act[client_nb,:,:] = activation
                    break

        print(clients_act.size())
        # Compute gm and reshape to match act size
        gm = compute_geometric_median(clients_act.cpu(), weights=None)
        gm = gm.median.to(self.device)
        gm = gm.unsqueeze(0)

        # Flatten act and gm to make them compatible for nn.CosineSimilarity
        act_flatten = clients_act.view(-1, self.activation_size)
        gm_flatten = gm.expand(len(clients), 250, self.activation_size).contiguous().view(-1, self.activation_size)

        # Compute distances and reshape to original shape (50, 10)
        cosine_dist = -1*self.cos_sim(act_flatten, gm_flatten).view(len(clients), -1)
        euclidean_dist = self.euclid_dist(act_flatten, gm_flatten).view(len(clients), -1)

        print("here shape gm", gm.shape,clients_act.shape, cosine_dist.shape)
        # Compute error 
        batch_err = cosine_dist * euclidean_dist 
        res = torch.mean(batch_err, dim=1)
        print("res", res.shape)

        return res.tolist()
    
    def filter_clients(self, selected_clients): 
        # Error computations and handling step
        print("okay")
        clients_re_np = np.array(self.compute_client_errors(selected_clients))
        valid_values = clients_re_np[np.isfinite(clients_re_np)]
        max_of_re = np.max(valid_values)
        mean_of_re = np.mean(valid_values)
        clients_re_without_nan = np.where(np.isnan(clients_re_np) | (clients_re_np == np.inf), max_of_re,
                                          clients_re_np)
        
        # Detection and filtering step
        selected_clients_array = np.array(selected_clients)
        good_updates = selected_clients_array[clients_re_without_nan < mean_of_re]
        for client in selected_clients_array[clients_re_without_nan >= mean_of_re]:
            client.suspect()
        
        # Compute new geomed from benign clients 
        # good_activations = self.activations[[not(client.is_suspect) for client in selected_clients]]
        # self.gm = compute_geometric_median(good_activations.cpu(), weights=None)

        self.summarize_detection(clients_re_without_nan, selected_clients_array, threshhold = mean_of_re)
        return good_updates
    
    def summarize_detection(self, reconstruction_errors, selected_clients, threshhold):
        selected_re = reconstruction_errors[reconstruction_errors < threshhold]
        selected_clients_info = selected_clients[reconstruction_errors < threshhold]
        selected_statuses = ["Attacker" if client.is_attacker else "Benign" for client in selected_clients_info]
        #sorted_selected_statuses = ["Attacker" if client.is_attacker else "Benign"
        #                            for _,client in sorted(zip(selected_re, selected_clients_info))]
        blocked_re = reconstruction_errors[reconstruction_errors >= threshhold]
        blocked_clients_info = selected_clients[reconstruction_errors >= threshhold]
        blocked_statuses = ["Attacker" if client.is_attacker else "Benign" for client in blocked_clients_info]
        #sorted_blocked_statuses = ["Attacker" if client.is_attacker else "Benign"
        #                           for _,client in sorted(zip(blocked_re, blocked_clients_info))]
        with open(os.path.join(self.dir_path,"filtering.txt"),"a+") as f :
            f.write(f"-----------------------\
            \nmean of re : {threshhold}\
            \nselected_client_errors : {selected_re}\
            \nselected_clients_status : {selected_statuses}\
            \nblocked_client_errors : {blocked_re}\
            \nblocked_clients_status : {blocked_statuses}")
            #\nselected_client_errors : {sorted(selected_re)}\
            #\nselected_clients_status : {sorted_selected_statuses}\
            #\nblocked_client_errors : {sorted(blocked_re)}\
            #\nblocked_clients_status : {sorted_blocked_statuses}")
        
        # Examine global model parameters
        with open(os.path.join(self.dir_path,"model_params.txt"),"a+") as f :
            f.write("---------\nnew epoch\n")
            for name, param in self.global_model.named_parameters():
                if param.requires_grad:
                    f.write(f"{name}, {param.data}\n")

        # Examine activations 
        with open(os.path.join(self.dir_path,"client_activations.txt"),"a+") as f :
            f.write("-------------\nnew round\n")
            # global model 
            #for data, label in self.trigger_loader:
            #    data, label = data.to(self.device), label.to(self.device)
            #    activation1 = self.global_model.get_activations_1(data) 
            #    activation2 = self.global_model.get_activations_2(data) 
            #    activation3 = self.global_model.get_activations_3(data) 
            #    activation4 = self.global_model.get_activations_4(data)
            #    f.write(f"global model activations\
            #          \n Activations 1 : max : {activation1.max()}, min : {activation1.min()}, avg {activation1.mean()}\
            #          \n Activations 2 : max : {activation2.max()}, min : {activation2.min()}, avg {activation2.mean()}\
            #          \n Activations 3 : max : {activation3.max()}, min : {activation3.min()}, avg {activation3.mean()}\
            #          \n Activations 4 : max : {activation4.max()}, min : {activation4.min()}, avg {activation4.mean()}\
            #          \n-------------\n")
            #    break
            ## clients
            for client_nb, client_model in enumerate(selected_clients):
                labels_cat = torch.tensor([]).to(self.device)
                for data, label in self.trigger_loader:
                    data, label = data.to(self.device), label.to(self.device)
                    activation1 = client_model.model.get_activations(data) 
                    f.write(f"Client nb : {client_nb}\
                    \n Attacker : {client_model.is_attacker}\
                    \n Activations 1 : max : {activation1.max()}, min : {activation1.min()}, avg {activation1.mean()}\n------------\n")
            #        activation1 = client_model.model.get_activations_1(data) 
            #        activation2 = client_model.model.get_activations_2(data) 
            #        activation3 = client_model.model.get_activations_3(data) 
            #        activation4 = client_model.model.get_activations_4(data)
            #        f.write(f"Client nb : {client_nb}\
            #              \n Attacker : {client_model.is_attacker}\
            #              \n Activations 1 : max : {activation1.max()}, min : {activation1.min()}, avg {activation1.mean()}\
            #              \n Activations 2 : max : {activation2.max()}, min : {activation2.min()}, avg {activation2.mean()}\
            #              \n Activations 3 : max : {activation3.max()}, min : {activation3.min()}, avg {activation3.mean()}\
            #              \n Activations 4 : max : {activation4.max()}, min : {activation4.min()}, avg {activation4.mean()}\
            #              \n-------------\n")
            #        break
