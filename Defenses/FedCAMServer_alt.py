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
        self.dir_path = f"Results/{experiment_name}/{self.dataset}/FedCAM/{self.cf['data_dist']}_{int(attacker_ratio * 100)}_{attack_type}"
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
    
    # Defense core functions
    def prepare_defense(self):
        if not self.cvae_trained:
            self.train_cvae()
            self.cvae_trained = True

    def compute_client_errors(self, clients): 
        return self.compute_reconstruction_error(clients)
    
    def filter_clients(self, selected_clients): 
        # Error computations and handling step
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

        # self.summarize_detection(clients_re_without_nan, selected_clients_array, threshhold = mean_of_re)
        return good_updates
    
    # Utils used by defense
    def train_cvae(self):
        if self.cvae_trained:
            print("CVAE is already trained, skipping re-training.")
            return

        init_ep = 10
        labels_act = 0
        input_models_act =  torch.zeros(size=(init_ep, self.activation_samples, self.activation_size)).to(self.device)
        input_cvae_model = deepcopy(self.global_model)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(input_cvae_model.parameters(), lr=self.cf["lr"], weight_decay=self.cf["wd"])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for epoch in range(init_ep):
            for data, labels in self.trigger_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = input_cvae_model(data)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                activation = input_cvae_model.get_activations(data)
                input_models_act[epoch] = activation
                labels_act = labels
                break
        
        gm = compute_geometric_median(input_models_act.cpu(), weights=None)
        input_models_act = input_models_act - gm.median.to(self.device)
        input_models_act = torch.sigmoid(input_models_act).detach()

        num_epochs = self.config_cvae["cvae_nb_ep"]
        optimizer = torch.optim.Adam(self.cvae.parameters(), lr=self.config_cvae["cvae_lr"],
                                     weight_decay=self.config_cvae["cvae_wd"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300], gamma=self.config_cvae["cvae_gamma"])

        for epoch in range(num_epochs):
            train_loss = 0
            loop = tqdm(input_models_act, leave=True)
            for batch_idx, activation in enumerate(loop):

                condition = Utils.one_hot_encoding(labels_act, self.num_classes, self.device)
                recon_batch, mu, logvar = self.cvae(activation, condition)
                loss = Utils.cvae_loss(recon_batch, activation, mu, logvar)

                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(loss=train_loss / (batch_idx + 1))

            scheduler.step()
        self.cvae_trained = True

    def compute_reconstruction_error(self, selected_clients):
        self.cvae.eval()

        clients_re = []

        clients_act = torch.zeros(size=(len(selected_clients), self.activation_samples, self.activation_size)).to(self.device)
        labels_cat = torch.tensor([]).to(self.device)

        for client_nb, client_model in enumerate(selected_clients):
            labels_cat = torch.tensor([]).to(self.device)
            for data, label in self.trigger_loader:
                data, label = data.to(self.device), label.to(self.device)
                activation = client_model.model.get_activations(data)
                clients_act[client_nb] = activation
                labels_cat = label
                break
        
        self.activations = clients_act
    
        gm = compute_geometric_median(clients_act.cpu(), weights=None)

        self.plot_geomed(selected_clients, clients_act, gm.median)

        clients_act = clients_act - gm.median.to(self.device)

        # clients_act = torch.abs(clients_act)
        clients_act = torch.sigmoid(clients_act)
        for client_act in clients_act:
            condition = Utils.one_hot_encoding(labels_cat, self.num_classes, self.device).to(self.device)
            recon_batch, _, _ = self.cvae(client_act, condition)
            mse = F.mse_loss(recon_batch, client_act, reduction='mean').item()
            clients_re.append(mse)

        return clients_re
    
    def plot_geomed(self, clients, acts, gm): 
        sns.set_theme()
        #fig, ax = plt.subplots()
        acts_df = pd.DataFrame(torch.mean(acts, dim=1).detach().cpu().numpy())   
        attacker_indexes = pd.Series([client.is_attacker for client in clients], index=acts_df.index)
        benign_indexes = pd.Series([not(client.is_attacker) for client in clients], index=acts_df.index)
        attacker_acts_df = acts_df[attacker_indexes].T
        benign_acts_df = acts_df[benign_indexes].T
        plot = benign_acts_df.plot(color = 'b', legend = False)
        attacker_acts_df.plot(ax = plot, color = 'r', legend = False)
        gm_values = gm.mean(dim=0).detach().tolist()
        plt.plot(gm_values, color = 'g')
        plt.savefig(os.path.join(self.dir_path,f"{datetime.datetime.now()}.png"))
    
    #def summarize_detection(self, reconstruction_errors, selected_clients, threshhold):
    #    selected_re = reconstruction_errors[reconstruction_errors < threshhold]
    #    selected_clients_info = selected_clients[reconstruction_errors < threshhold]
    #    selected_statuses = ["Attacker" if client.is_attacker else "Benign" for client in selected_clients_info]
    #    #sorted_selected_statuses = ["Attacker" if client.is_attacker else "Benign"
    #    #                            for _,client in sorted(zip(selected_re, selected_clients_info))]
    #    blocked_re = reconstruction_errors[reconstruction_errors >= threshhold]
    #    blocked_clients_info = selected_clients[reconstruction_errors >= threshhold]
    #    blocked_statuses = ["Attacker" if client.is_attacker else "Benign" for client in blocked_clients_info]
    #    #sorted_blocked_statuses = ["Attacker" if client.is_attacker else "Benign"
    #    #                           for _,client in sorted(zip(blocked_re, blocked_clients_info))]
    #    with open(os.path.join(self.dir_path,"filtering.txt"),"a+") as f :
    #        f.write(f"-----------------------\
    #        \nmean of re : {threshhold}\
    #        \nselected_client_errors : {selected_re}\
    #        \nselected_clients_status : {selected_statuses}\
    #        \nblocked_client_errors : {blocked_re}\
    #        \nblocked_clients_status : {blocked_statuses}")
    #        #\nselected_client_errors : {sorted(selected_re)}\
    #        #\nselected_clients_status : {sorted_selected_statuses}\
    #        #\nblocked_client_errors : {sorted(blocked_re)}\
    #        #\nblocked_clients_status : {sorted_blocked_statuses}")
    #    
    #    # Examine global model parameters
    #    with open(os.path.join(self.dir_path,"model_params.txt"),"a+") as f :
    #        f.write("---------\nnew epoch\n")
    #        for name, param in self.global_model.named_parameters():
    #            if param.requires_grad:
    #                f.write(f"{name}, {param.data}\n")
#
    #    # Examine activations 
    #    with open(os.path.join(self.dir_path,"client_activations.txt"),"a+") as f :
    #        f.write("-------------\nnew round\n")
    #        # global model 
    #        #for data, label in self.trigger_loader:
    #        #    data, label = data.to(self.device), label.to(self.device)
    #        #    activation1 = self.global_model.get_activations_1(data) 
    #        #    activation2 = self.global_model.get_activations_2(data) 
    #        #    activation3 = self.global_model.get_activations_3(data) 
    #        #    activation4 = self.global_model.get_activations_4(data)
    #        #    f.write(f"global model activations\
    #        #          \n Activations 1 : max : {activation1.max()}, min : {activation1.min()}, avg {activation1.mean()}\
    #        #          \n Activations 2 : max : {activation2.max()}, min : {activation2.min()}, avg {activation2.mean()}\
    #        #          \n Activations 3 : max : {activation3.max()}, min : {activation3.min()}, avg {activation3.mean()}\
    #        #          \n Activations 4 : max : {activation4.max()}, min : {activation4.min()}, avg {activation4.mean()}\
    #        #          \n-------------\n")
    #        #    break
    #        ## clients
    #        for client_nb, client_model in enumerate(selected_clients):
    #            labels_cat = torch.tensor([]).to(self.device)
    #            for data, label in self.trigger_loader:
    #                data, label = data.to(self.device), label.to(self.device)
    #                activation1 = client_model.model.get_activations(data) 
    #                f.write(f"Client nb : {client_nb}\
    #                \n Attacker : {client_model.is_attacker}\
    #                \n Activations 1 : max : {activation1.max()}, min : {activation1.min()}, avg {activation1.mean()}\n------------\n")
    #        #        activation1 = client_model.model.get_activations_1(data) 
    #        #        activation2 = client_model.model.get_activations_2(data) 
    #        #        activation3 = client_model.model.get_activations_3(data) 
    #        #        activation4 = client_model.model.get_activations_4(data)
    #        #        f.write(f"Client nb : {client_nb}\
    #        #              \n Attacker : {client_model.is_attacker}\
    #        #              \n Activations 1 : max : {activation1.max()}, min : {activation1.min()}, avg {activation1.mean()}\
    #        #              \n Activations 2 : max : {activation2.max()}, min : {activation2.min()}, avg {activation2.mean()}\
    #        #              \n Activations 3 : max : {activation3.max()}, min : {activation3.min()}, avg {activation3.mean()}\
    #        #              \n Activations 4 : max : {activation4.max()}, min : {activation4.min()}, avg {activation4.mean()}\
    #        #              \n-------------\n")
    #        #        break
#