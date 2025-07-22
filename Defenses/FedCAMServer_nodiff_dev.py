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
        self.dir_path = f"Results/{experiment_name}/{self.dataset}/FedCAM_dev/{self.cf['data_dist']}_{int(attacker_ratio * 100)}_{attack_type}"
        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)
        os.makedirs(self.dir_path)

        # Saving directory for plots 
        for plot_type in ["gm_all_layers","cvae_latent_space","gm_per_layer","failed_reconstruction","reconstructed_successfully"] :
            os.makedirs(os.path.join(self.dir_path,plot_type))

        # Defense parameters
        self.activation_size = cf["cvae_input_dim"]
        self.activation_samples = cf["FEMNIST_num_clients_trigger"] if self.dataset == "FEMNIST" else cf["size_trigger"]
        self.num_classes = cf["num_classes"]
        self.config_cvae = {"cvae_nb_ep": cf["cvae_nb_ep"],
                            "cvae_lr": cf["cvae_lr"],
                            "cvae_wd": cf["cvae_wd"],
                            "cvae_gamma": cf["cvae_gamma"],}
        self.gm_memory = cf["gm_memory_ratio"]
        
        # Initiate CVAE
        self.cvae_trained = False
        self.cvae = CVAE(input_dim=cf["cvae_input_dim"],
            condition_dim=self.cf["condition_dim"],
            hidden_dim=self.cf["hidden_dim"],
            latent_dim=self.cf["latent_dim"]).to(self.device)
        
        # Storage for geomed computation 
        self.stored_gm = None 
    
    # Defense core functions
    def prepare_defense(self):
        # Train CVAE
        if self.defence:
            if not self.cvae_trained:
                self.train_cvae()
                self.cvae_trained = True

    def compute_client_errors(self, clients): 
        return self.compute_reconstruction_error(clients)
    
    def filter_clients(self, selected_clients): 
        # Error computations and handling step
        clients_re, clients_re_without_cvae = self.compute_client_errors(selected_clients)
        selected_clients_array = np.array(selected_clients)

        # With CVAE
        clients_re_np = np.array(clients_re)
        valid_values = clients_re_np[np.isfinite(clients_re_np)]
        cvae_max_of_re = np.max(valid_values)
        cvae_mean_of_re = np.mean(valid_values)
        cvae_clients_re_without_nan = np.where(np.isnan(clients_re_np) | (clients_re_np == np.inf), cvae_max_of_re,
                                          clients_re_np)
        cvae_good_updates = selected_clients_array[cvae_clients_re_without_nan < cvae_mean_of_re]
        cvae_blocked = selected_clients_array[cvae_clients_re_without_nan >= cvae_mean_of_re]
        
        # Without CVAE
        # clients_re_np = np.array(clients_re_without_cvae)
        # valid_values = clients_re_np[np.isfinite(clients_re_np)]
        # no_cvae_max_of_re = np.max(valid_values)
        # no_cvae_mean_of_re = np.mean(valid_values)
        # no_cvae_clients_re_without_nan = np.where(np.isnan(clients_re_np) | (clients_re_np == np.inf), no_cvae_max_of_re,
                                        #   clients_re_np)
        # no_cvae_good_updates = selected_clients_array[no_cvae_clients_re_without_nan < no_cvae_mean_of_re]
        # no_cvae_blocked = selected_clients_array[no_cvae_clients_re_without_nan >= no_cvae_mean_of_re]

        # if np.array_equal(cvae_good_updates, no_cvae_good_updates) : 
            # print("CVAE and NO CVAE are the same")
        # else :
            # print("Detection is different when adding CVAE this round")
            # print(f"With CVAE : \
            # \nBlocked {sum([client.is_attacker for client in cvae_blocked])}/{sum([client.is_attacker for client in selected_clients])} attackers \
            # \nBlocked {sum([not(client.is_attacker) for client in cvae_blocked])} benign clients ")
            # print(f"Without CVAE : \
            # \nBlocked {sum([client.is_attacker for client in no_cvae_blocked])}/{sum([client.is_attacker for client in selected_clients])} attackers \
            # \nBlocked {sum([not(client.is_attacker) for client in no_cvae_blocked])} benign clients ")

        # self.summarize_detection(clients_re_without_nan, selected_clients_array, threshhold = mean_of_re)
        return cvae_good_updates
    
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
        
        with torch.no_grad(): 
            gm = compute_geometric_median(input_models_act.cpu(), weights=None)
            print(gm.termination)

        # input_models_act = input_models_act - gm.median.to(self.device)
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
        clients_re_without_cvae = []

        clients_act = torch.zeros(size=(len(selected_clients), self.activation_samples, self.activation_size)).to(self.device)
        clients_lvs = torch.zeros(size=(len(selected_clients), self.activation_samples, self.cf["latent_dim"])).to(self.device)
        labels_cat = torch.tensor([]).to(self.device)

        for client_nb, client_model in enumerate(selected_clients):
            labels_cat = torch.tensor([]).to(self.device)
            for data, label in self.trigger_loader:
                data, label = data.to(self.device), label.to(self.device)
                activation = client_model.model.get_activations(data)
                clients_act[client_nb] = activation
                labels_cat = label
                break
        
        with torch.no_grad() : # Encountered memory leakage here otherwise. Is it because pytorch saves computation graphs for gm ????
            gm = compute_geometric_median(clients_act.cpu(), weights=None)
            print(gm.termination)
            if self.stored_gm : 
                gm.median = self.gm_memory*self.stored_gm.median + (1-self.gm_memory)*gm.median

        # No CVAE
        # for client_act in clients_act:
            # mse = F.mse_loss(gm.median.to(self.device), client_act, reduction='mean').item()
            # clients_re_without_cvae.append(mse)

        # Plot geomed and activations
        self.plot_geomed(selected_clients, clients_act, gm.median)
        self.plot_acts_gm_per_layer(selected_clients, clients_act, gm.median)

        # With CVAE
        # clients_act = clients_act - gm.median.to(self.device)
        # clients_act = torch.abs(clients_act)
        clients_act = torch.sigmoid(clients_act)
        for client_act in clients_act:
            condition = Utils.one_hot_encoding(labels_cat, self.num_classes, self.device).to(self.device)
            recon_batch, _, _ = self.cvae(client_act, condition)
            recon_batch_bis, _, _ = self.cvae(client_act, condition)
            print(f"Distance between two draws : {F.l1_loss(recon_batch, recon_batch_bis, reduction='mean').item()}")
            mse = F.mse_loss(recon_batch, client_act, reduction='mean').item()
            clients_re.append(mse)

        #self.stored_gm = gm
        
        # Plot activation representations (after geomed diff)
        for client_nb, client_act in enumerate(clients_act) : 
            condition = Utils.one_hot_encoding(labels_cat, self.num_classes, self.device).to(self.device)
            clients_lvs[client_nb] = self.cvae.get_latent_repr(client_act, condition)
        self.plot_latent_vector(selected_clients, clients_lvs, labels_cat, title = "after diff with geomed")

        # Plot highest and lowest re client reconstructed vectors
        highest, lowest = np.argmax(clients_re), np.argmin(clients_re)
        self.plot_reconstructed_vs_real(clients_act[highest], recon_batch[highest], clients_re[highest],
                                        failed = True, benign = not(selected_clients[highest].is_attacker))
        self.plot_reconstructed_vs_real(clients_act[lowest], recon_batch[lowest], clients_re[lowest],
                                        failed = False, benign = not(selected_clients[lowest].is_attacker))
        
        # Compute distances in cvae output
        for client_nb in range(len(selected_clients)-1): 
            print(f"Distance between {selected_clients[client_nb].is_attacker} and {selected_clients[client_nb+1].is_attacker} : \
                  {F.l1_loss(recon_batch[client_nb], recon_batch[client_nb+1], reduction='mean').item()}")

        # Compute new geomed from benign clients 
        # good_activations = clients_act[[not(client.is_suspect) for client in selected_clients]]
        # self.stored_gm = compute_geometric_median(good_activations.cpu(), weights=None)

        return clients_re, clients_re_without_cvae
    
    def compute_activations(self, selected_clients): 

        acts = torch.zeros(size=(len(selected_clients), self.activation_samples, self.activation_size)).to(self.device)
        labels_cat = torch.tensor([]).to(self.device)

        with torch.no_grad():
            for client_nb, client_model in enumerate(selected_clients):
                labels_cat = torch.tensor([]).to(self.device)
                for data, label in self.trigger_loader:
                    data, label = data.to(self.device), label.to(self.device)
                    activation = client_model.model.get_activations(data)
                    acts[client_nb] = activation
                    labels_cat = label
                    break
        
        return acts
    
    def summarize_detection(self, reconstruction_errors, selected_clients, threshhold):
        selected_re = reconstruction_errors[reconstruction_errors < threshhold]
        selected_clients_info = selected_clients[reconstruction_errors < threshhold]
        selected_statuses = ['Attacker' if client.is_attacker else 'Benign' for client in selected_clients_info]
        #sorted_selected_statuses = ['Attacker' if client.is_attacker else 'Benign'
        #                            for _,client in sorted(zip(selected_re, selected_clients_info))]
        blocked_re = reconstruction_errors[reconstruction_errors >= threshhold]
        blocked_clients_info = selected_clients[reconstruction_errors >= threshhold]
        blocked_statuses = ['Attacker' if client.is_attacker else 'Benign' for client in blocked_clients_info]
        #sorted_blocked_statuses = ['Attacker' if client.is_attacker else 'Benign'
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
            for client_nb, client_model in enumerate(selected_clients):
                labels_cat = torch.tensor([]).to(self.device)
                for data, label in self.trigger_loader:
                    data, label = data.to(self.device), label.to(self.device)
                    activation1 = client_model.model.get_activations(data) 
                    f.write(f"Client nb : {client_nb}\
                    \n Attacker : {client_model.is_attacker}\
                    \n Activations 1 : max : {activation1.max()}, min : {activation1.min()}, avg {activation1.mean()}\n------------\n")
    
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
        plt.savefig(os.path.join(self.dir_path,"gm_all_layers",f"{datetime.datetime.now()}.png"))
        plt.close()
    
    def plot_latent_vector(self, clients, latent_vectors, conditions, title): 
        sns.set_theme()
        #fig, ax = plt.subplots()
        # Generate latent vector dataframes
        latent_vectors_df = pd.DataFrame(latent_vectors.view(-1,self.cf["latent_dim"]).detach().cpu().numpy())   
        attacker_indexes = pd.Series([client.is_attacker for client in clients] * 250, index=latent_vectors_df.index)
        benign_indexes = pd.Series([not(client.is_attacker) for client in clients] * 250, index=latent_vectors_df.index)
        attacker_latent_vectors_df = latent_vectors_df[attacker_indexes]
        benign_latent_vectors_df = latent_vectors_df[benign_indexes]
        nb_attackers = attacker_latent_vectors_df.shape[0] // 250
        nb_benign = benign_latent_vectors_df.shape[0] // 250
        attacker_conditions = list(conditions.cpu().numpy()) * nb_attackers
        benign_conditions = list(conditions.cpu().numpy()) * nb_benign
        # Generate figure wwith plots for each label 
        fig, axs = plt.subplots(5,2, figsize =(50,50)) 
        # Plot each label latent space 
        for idx, ax in enumerate(axs.reshape(-1)): 
            ax.set_title(f"Label = {idx}") 
            attacker_label_mask = (np.array(attacker_conditions) == idx) 
            benign_label_mask = (np.array(benign_conditions) == idx) 
            sns.stripplot(benign_latent_vectors_df[benign_label_mask],ax = ax, color = 'b', legend = False, orient='h')
            sns.stripplot(attacker_latent_vectors_df[attacker_label_mask],ax = ax, color = 'r', legend = False, orient='h')
        plt.savefig(os.path.join(self.dir_path,"cvae_latent_space",f"{title}_{datetime.datetime.now()}.png"))
        plt.close()

    def plot_acts_gm_per_layer(self, clients, acts, gm): 
        sns.set_theme() 
        # Prepare data 
        acts_df = pd.DataFrame(torch.mean(acts, dim=1).detach().cpu().numpy())   
        attacker_indexes = pd.Series([client.is_attacker for client in clients], index=acts_df.index)
        benign_indexes = pd.Series([not(client.is_attacker) for client in clients], index=acts_df.index)
        attacker_acts_df = acts_df[attacker_indexes].T
        benign_acts_df = acts_df[benign_indexes].T
        gm_values = gm.mean(dim=0).detach().tolist()
        # Layer split indexes 
        layersizes = [0,18432, 21632, 22208, 22464]
        # Create figures
        fig, axs = plt.subplots(2,2, figsize =(50,50)) 
        for idx, ax in enumerate(axs.reshape(-1)): 
            layer = slice(layersizes[idx],layersizes[idx+1])
            ax.set_title(f"Layer {idx}") 
            benign_data = benign_acts_df.iloc[layer,:]
            benign_data.reset_index(inplace =True, drop = True)
            benign_data.plot(ax = ax, color = 'b', legend = False)
            attacker_data = attacker_acts_df.iloc[layer,:]
            attacker_data.reset_index(inplace = True, drop = True)
            attacker_data.plot(ax = ax, color = 'r', legend = False)
            ax.plot(gm_values[layer], color = 'g')
        plt.savefig(os.path.join(self.dir_path,"gm_per_layer",f"{datetime.datetime.now()}.png"))
        plt.close()

    def plot_reconstructed_vs_real(self, acts, reconstructed_acts, reconstruction_error, failed = False, benign = True): 
        sns.set_theme()
        color = 'b' if benign else 'r'
        savepath = "failed_reconstruction" if failed else "reconstructed_successfully"
        # Layers split
        layersizes = [0,18432, 21632, 22208, 22464]
        acts, reconstructed_acts = acts.detach().cpu().numpy().mean(axis=0), reconstructed_acts.detach().cpu().numpy()
        print(reconstructed_acts.shape)
        # Create figures
        fig, axs = plt.subplots(4,2, figsize =(50,50)) 
        fig.suptitle(f"reconstruction error : {reconstruction_error}")
        # Plot real acts 
        for idx in range(len(layersizes)-1): 
            ax = axs[idx,0]
            ax.set_title(f"layer {idx}")
            ax.plot(acts[layersizes[idx]:layersizes[idx+1]], color = color)
        # Plot reconstructed acts 
        for idx in range(len(layersizes)-1): 
            ax = axs[idx,1]
            ax.set_title(f"layer {idx} (reconstructed)")
            ax.plot(reconstructed_acts[layersizes[idx]:layersizes[idx+1]], color = color)
        # Save figure 
        plt.savefig(os.path.join(self.dir_path,savepath,f"{datetime.datetime.now()}.png"))
        plt.close()

    def plot_acts_over_time(self, acts_hist, layername): 
        pass 



