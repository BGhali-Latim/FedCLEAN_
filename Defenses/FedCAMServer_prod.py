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
from math import sqrt, floor
import skimage.util as skimg
from sklearn.cluster import HDBSCAN

from Server.Server import Server

class DefenseServer(Server):
    def __init__(self, cf=None, model=None, attack_type=None, attacker_ratio=None, dataset = None, experiment_name = 'debug'):
        super().__init__(cf, model, attack_type, attacker_ratio, dataset)

        # Saving directory
        self.dir_path = f"Results/{experiment_name}/{self.dataset}/FedCAM_dev/{self.cf['data_dist']}_{int(attacker_ratio * 100)}_{attack_type}"
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
        self.gm_memory = cf["gm_memory_ratio"]
        self.retrain_milestone = cf["cvae_retrain_epochs"]
        self.with_retraining = cf["retrain_cvae"]
        
        # Initiate CVAE
        self.cvae_trained = False
        self.cvae = CVAE(input_dim=cf["cvae_input_dim"],
            condition_dim=self.cf["condition_dim"],
            hidden_dim=self.cf["hidden_dim"],
            latent_dim=self.cf["latent_dim"]).to(self.device)
        
        # Storage 
        self.stored_gm = None 
        self.cvae_last_trained = 0

        # Metric storage 
        self.attacker_lower_bounds = [] 
        self.attacker_upper_bounds = []
        self.benign_lower_bounds = [] 
        self.benign_upper_bounds = [] 
        self.decision_boundaries = [] 
        self.decision_boundaries_weighted = []
    
    # Defense core functions
    def prepare_defense(self):
        # Train CVAE
        if self.defence:
            if not self.cvae_trained:
                self.train_cvae()
                self.cvae_trained = True

    def compute_client_errors(self, clients): 
        self.cvae_last_trained += 1 
        if self.cvae_last_trained % self.retrain_milestone == 0 :
            print("Retraining cvae on updated global model")
            if self.with_retraining :
                self.retrain_cvae()
        return self.compute_reconstruction_error(clients)
    
    def filter_clients(self, selected_clients): 
        # Error computations and handling step
        clients_re= self.compute_client_errors(selected_clients)
        selected_clients_array = np.array(selected_clients)

        # With CVAE
        clients_re_np = np.array(clients_re)
        valid_values = clients_re_np[np.isfinite(clients_re_np)]
        max_of_re = np.max(valid_values)
        mean_of_re = np.mean(valid_values)
        self.decision_boundaries.append(mean_of_re)
        self.decision_boundaries_weighted.append(2*mean_of_re)
        # mean_of_re = 2* mean_of_re # Change back here
        clients_re_without_nan = np.where(np.isnan(clients_re_np) | (clients_re_np == np.inf), max_of_re,
                                          clients_re_np)
        good_updates = selected_clients_array[clients_re_without_nan < mean_of_re]

        # If using hdbscan 
        cvae_good_update_indexes = self.get_clusters(clients_re_without_nan)
        good_updates = selected_clients_array[cvae_good_update_indexes]

        return good_updates
    
    def compute_reconstruction_error(self, selected_clients):
            self.cvae.eval()
    
            clients_re = []
            clients_energies = []
    
            # Real client identities (this is only used for visualisation, do not leak this into detection !!!!!!!!!!!)
            attacker_indexes = [i for i in range(len(selected_clients)) if selected_clients[i].is_attacker]
            benign_indexes = [i for i in range(len(selected_clients)) if not(selected_clients[i].is_attacker)]
    
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
    
            # With CVAE
            clients_act = clients_act - gm.median.to(self.device)
            clients_act = torch.sigmoid(clients_act)
            for client_act in clients_act:
                condition = Utils.one_hot_encoding(labels_cat, self.num_classes, self.device).to(self.device)
                recon_batch, _, _ = self.cvae(client_act, condition)
                # recon_batch_bis, _, _ = self.cvae(client_act, condition)
                # print(f"Distance between two draws : {F.l1_loss(recon_batch, recon_batch_bis, reduction='mean').item()}")
                mse = F.mse_loss(recon_batch, client_act, reduction='mean').item()
                clients_energies.append(torch.mean(torch.square(client_act)).item())
                clients_re.append(mse)
            
            # Compute new geomed from benign clients 
            # good_activations = clients_act[[not(client.is_suspect) for client in selected_clients]]
            # self.stored_gm = compute_geometric_median(good_activations.cpu(), weights=None)
            #self.stored_gm = gm
    
            return clients_re
    
    # Utils used by defense
    def compute_activations(self, selected_clients): 
        acts = torch.zeros(size=(len(selected_clients), self.activation_samples, self.activation_size)).to(self.device)
        with torch.no_grad():
            for client_nb, client_model in enumerate(selected_clients):
                for data, label in self.trigger_loader:
                    data, label = data.to(self.device), label.to(self.device)
                    activation = client_model.model.get_activations(data)
                    acts[client_nb] = activation
                    break
        
        return acts
        
    def train_cvae(self):
        if self.cvae_trained:
            print("CVAE is already trained, skipping re-training.")
            return

        init_ep = 10
        warmup_ep = 10 
        labels_act = 0
        input_models_act =  torch.zeros(size=(init_ep, self.activation_samples, self.activation_size)).to(self.device)
        input_cvae_model = deepcopy(self.global_model)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(input_cvae_model.parameters(), lr=self.cf["lr"], weight_decay=self.cf["wd"])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for epoch in range(warmup_ep):
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
                #break

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
                #break
        # 
        with torch.no_grad(): 
            gm = compute_geometric_median(input_models_act.cpu(), weights=None)
            print(gm.termination)

        input_models_act = input_models_act - gm.median.to(self.device)
        input_models_act = torch.sigmoid(input_models_act).detach()

        num_epochs = self.config_cvae["cvae_nb_ep"]
        optimizer = torch.optim.Adam(self.cvae.parameters(), lr=self.config_cvae["cvae_lr"],
                                     weight_decay=self.config_cvae["cvae_wd"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=self.config_cvae["cvae_gamma"])

        mse_over_epochs = []
        KL_over_epochs = []
        loss_over_epochs = []

        for epoch in range(num_epochs):
            train_loss = 0
            epoch_mse = 0
            epoch_kld = 0
            loop = tqdm(input_models_act, leave=True)
            last_batch_idx = 0
            for batch_idx, activation in enumerate(loop):

                condition = Utils.one_hot_encoding(labels_act, self.num_classes, self.device)
                recon_batch, mu, logvar = self.cvae(activation, condition)
                mse, kld = Utils.cvae_loss(recon_batch, activation, mu, logvar)

                # match self.cvae_loss :
                    # case "base" : 
                        # loss = mse + kld 
                    # case "weighted" : 
                        # loss = mse + 0.5 * kld 
                    # case "annealed" : 
                        # loss = mse + (0.5 * (num_epochs//10)) * kld 
                    # case "calibrated" : 
                        # log_sigma_opt = 0.5 * mse.log() 
                        # r_loss = 0.5 * (mse / torch.pow(log_sigma_opt.exp(),2)) + log_sigma_opt
                        # r_loss = log_sigma_opt + 0.5 * (mse/ log_sigma_opt.exp().pow(2))
                        # r_loss = r_loss.sum()
                        # loss = r_loss + kld

                loss = mse + (0.5 * (num_epochs//10)) * kld

                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                epoch_mse += mse.item()
                epoch_kld += kld.item()
                optimizer.step()
                last_batch_idx = batch_idx
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(dkl=epoch_kld / (last_batch_idx + 1),
                                 mse=epoch_mse / (last_batch_idx + 1),
                                 loss=train_loss / (last_batch_idx + 1))
                
            mse_over_epochs.append(epoch_mse / (last_batch_idx + 1))
            KL_over_epochs.append(epoch_kld / (last_batch_idx + 1))
            loss_over_epochs.append(train_loss / (last_batch_idx + 1))
            scheduler.step()
        self.plot_cvae_loss(mse_over_epochs, KL_over_epochs, loss_over_epochs)
        self.cvae_trained = True
    
    def retrain_cvae(self):
        init_ep = 10
        warmup_ep = 0 
        labels_act = 0
        input_models_act =  torch.zeros(size=(init_ep, self.activation_samples, self.activation_size)).to(self.device)
        input_cvae_model = deepcopy(self.global_model)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(input_cvae_model.parameters(), lr=self.cf["lr"], weight_decay=self.cf["wd"])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for epoch in range(warmup_ep):
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
                #break
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
                #break
        # 
        with torch.no_grad(): 
            gm = compute_geometric_median(input_models_act.cpu(), weights=None)
            print(gm.termination)
        input_models_act = input_models_act - gm.median.to(self.device)
        input_models_act = torch.sigmoid(input_models_act).detach()
        num_epochs = self.config_cvae["cvae_nb_ep"]
        optimizer = torch.optim.Adam(self.cvae.parameters(), lr=self.config_cvae["cvae_lr"],
                                     weight_decay=self.config_cvae["cvae_wd"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=self.config_cvae["cvae_gamma"])
        mse_over_epochs = []
        KL_over_epochs = []
        loss_over_epochs = []
        for epoch in range(num_epochs):
            train_loss = 0
            epoch_mse = 0
            epoch_kld = 0
            loop = tqdm(input_models_act, leave=True)
            last_batch_idx = 0
            for batch_idx, activation in enumerate(loop):
                condition = Utils.one_hot_encoding(labels_act, self.num_classes, self.device)
                recon_batch, mu, logvar = self.cvae(activation, condition)
                mse, kld = Utils.cvae_loss(recon_batch, activation, mu, logvar)
                loss = mse + (0.5 * (num_epochs//10)) * kld
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                epoch_mse += mse.item()
                epoch_kld += kld.item()
                optimizer.step()
                last_batch_idx = batch_idx
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(dkl=epoch_kld / (last_batch_idx + 1),
                                 mse=epoch_mse / (last_batch_idx + 1),
                                 loss=train_loss / (last_batch_idx + 1))
                
            mse_over_epochs.append(epoch_mse / (last_batch_idx + 1))
            KL_over_epochs.append(epoch_kld / (last_batch_idx + 1))
            loss_over_epochs.append(train_loss / (last_batch_idx + 1))
            scheduler.step()
        self.plot_cvae_loss(mse_over_epochs, KL_over_epochs, loss_over_epochs)
        self.cvae_trained = True
        
    def get_clusters(self, reconstruction_errors): 
        hdb = HDBSCAN(min_cluster_size = 8)
        highest_re_index = np.argmax(reconstruction_errors)
        # HDBSCAN
        reconstruction_errors = reconstruction_errors.reshape(-1,1)
        hdb.fit(reconstruction_errors)
        dbscan_result = hdb.labels_
        dbscan_clusters = np.unique(dbscan_result)
        print(f"HDBSCAN found {len(dbscan_clusters)} clusters")
        # Filtering
        cluster_ceilings = []
        for cluster_id in dbscan_clusters:
            # get data points that fall in this cluster
            cluster_indexes = np.array([dbscan_result == cluster_id]).T
            cluster = reconstruction_errors[cluster_indexes]
            cluster_ceilings.append(max(cluster))
        # Logging
        attackers_cluster_id = dbscan_result[highest_re_index]
        self.plot_clusters(reconstruction_errors, dbscan_result, sorted(cluster_ceilings), attackers_cluster_id)
        # get the DBSCAN clusters
        attackers_cluster_id = dbscan_result[highest_re_index]
        return(dbscan_result != attackers_cluster_id)
        #return(reconstruction_errors.flatten() <= sorted(cluster_ceilings)[-2])
    
    def trust_propagation(self, selected_clients, clients_re):
        eps = 0.1
        sorted_clients, sorted_re = [client for _, client in sorted(zip(clients_re, selected_clients))], sorted(clients_re)
        threshold = (max(clients_re) - min(clients_re)) * eps
        good_updates = [sorted_clients[0]] 
        for idx,client in enumerate(sorted_clients[:-1]) :
            if (sorted_re[idx+1]-sorted_re[idx]) > threshold : # If next has much higher re, is probably an attacker
                break 
            good_updates.append(sorted_clients[idx+1])
        return(good_updates)
    
    def plot_clusters(self, errors, clusters, decision_boundaries, attacker_cluster):
        fig, ax = plt.subplots()
        cluster_ids = np.unique(clusters)
        ax.set(title = f"nb_clusters = {len(cluster_ids)}")
        for cluster_id in cluster_ids:
            # get data points that fall in this cluster
            cluster_indexes = np.array([clusters == cluster_id]).T
            cluster = errors[cluster_indexes]
            sns.stripplot(cluster, ax = ax, orient = 'h')
        #for boundary in decision_boundaries : 
        #    ax.axvline(x = boundary, linestyle = '--')
        ax.axvline(x=errors.mean(), linestyle = '--', color = 'g')
        #ax.axvline(x=decision_boundaries[-2], linestyle = '--', color = 'y')
        plt.xlabel("Reconstruction error")
        plt.savefig(os.path.join(self.dir_path,"re_filtering",f"{datetime.datetime.now()}.png"))
        plt.close()
    
    def plot_trust_prop(self, errors, good_updates):
        fig, ax = plt.subplots()
        errors_sorted = sorted(errors)
        ax.set(title = f"Clients selected by trust propagation")
        sns.stripplot(errors_sorted[:len(good_updates)], ax = ax, orient = 'h', color = 'b')
        sns.stripplot(errors_sorted[len(good_updates):], ax = ax, orient = 'h', color = 'r')
        ax.axvline(x=errors.mean(), linestyle = '--', color = 'g')
        #ax.axvline(x=decision_boundaries[-2], linestyle = '--', color = 'y')
        plt.xlabel("Reconstruction error")
        plt.savefig(os.path.join(self.dir_path,"re_filtering",f"{datetime.datetime.now()}.png"))
        plt.close()
    
    def plot_cvae_loss(self, mse_list, kld_list, loss_list): 
        sns.set_theme()
        # Create figures
        fig, ax = plt.subplots(1,1, figsize =(20,20)) 
        #fig.suptitle(f"reconstruction error : {reconstruction_error}")
        # Plot real acts 
        ax.plot(kld_list, color = 'orange', label ='Kullback-Liebler Divergence')
        ax.plot(mse_list, color = 'b', label ='Reconstruction error (MSE)')
        ax.plot(loss_list, color = 'g', linestyle = '--', label = "total cvae loss")
        plt.yscale("log")
        ax.legend()
        ax.set(title = "CVAE loss components over training epochs")
        plt.xlabel("CVAE training epochs")
        plt.ylabel("Loss value (log)")
        plt.ylim(10e-8 ,10e+4)
        # Save figure 
        plt.savefig(os.path.join(self.dir_path,"CVAE_training.png"))
        #plt.show()
        plt.close()



