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
from sklearn.cluster import HDBSCAN, DBSCAN, AffinityPropagation, MeanShift
from sklearn.decomposition import PCA
from scipy.stats import norm

from Server.Server import Server

class DefenseServer(Server):
    def __init__(self, cf=None, model=None, attack_type=None, attacker_ratio=None, dataset = None, sampler = None, experiment_name = 'debug', lamda = 0.2):
        super().__init__(cf, model, attack_type, attacker_ratio, dataset, sampler)
        # Defense server
        self.defence = True

        # Saving directory
        self.dir_path = f"Results/{experiment_name}/{self.dataset}/{sampler.name}/FedCAM_dev/{self.cf['data_dist']}_{int(attacker_ratio * 100)}_{attack_type}"
        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)
        os.makedirs(self.dir_path)

        # Saving directory for plots 
        #for plot_type in ["gm_all_layers","cvae_latent_space","gm_per_layer","failed_reconstruction","reconstructed_successfully", "re_filtering"] :
        #    os.makedirs(os.path.join(self.dir_path,plot_type))

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
        self.lamda = lamda
        
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
        """ self.attacker_lower_bounds = [] 
        self.attacker_upper_bounds = []
        self.benign_lower_bounds = [] 
        self.benign_upper_bounds = [] 
        self.decision_boundaries = [] 
        self.decision_boundaries_weighted = [] """
    
    # Defense core functions
    def prepare_defense(self):
        # Train CVAE
        if self.defence:
            if not self.cvae_trained:
                self.train_cvae()
                self.cvae_trained = True

    def compute_client_errors(self, clients): 
        #self.cvae_last_trained += 1 
        #if self.cvae_last_trained % self.retrain_milestone == 0 :
        #    self.plot_decision_boundaries()
        #    print("Retraining cvae on updated global model")
        #    if self.with_retraining :
        #        self.retrain_cvae()
        return self.compute_reconstruction_error(clients)
    
    def filter_clients(self, selected_clients): 
        # Error computations and handling step
        clients_re = self.compute_client_errors(selected_clients)
        # lowest_re_client_redraw = clients_redraw_err[np.argmin(clients_re)]
        selected_clients_array = np.array(selected_clients)

        # With CVAE
        clients_re_np = np.array(clients_re)
        valid_values = clients_re_np[np.isfinite(clients_re_np)]
        max_of_re = np.max(valid_values)
        mean_of_re = np.mean(valid_values)
        #self.decision_boundaries.append(mean_of_re)
        # mean_of_re = 2* mean_of_re # Change back here
        clients_re_without_nan = np.where(np.isnan(clients_re_np) | (clients_re_np == np.inf), max_of_re,
                                          clients_re_np)
        # good_updates = selected_clients_array[clients_re_without_nan < mean_of_re]

        # If using hdbscan 
        # cvae_good_update_indexes = self.get_clusters(clients_re_without_nan)
        # good_updates = selected_clients_array[cvae_good_update_indexes]

        # If propagation 
        # print(selected_clients_array)
        # print(clients_re_without_nan)
        good_updates = self.trust_propagation(selected_clients_array, clients_re_without_nan)
        #self.plot_trust_prop(clients_re_without_nan, good_updates)

        return good_updates
    
    def compute_reconstruction_error(self, selected_clients):
            self.cvae.eval()
    
            clients_re = []
            # clients_energies = []
            # clients_redraw_errors = []
    
            clients_act = torch.zeros(size=(len(selected_clients), self.activation_samples, self.activation_size)).to(self.device)
            clients_lvs = torch.zeros(size=(len(selected_clients), self.activation_samples, self.cf["latent_dim"])).to(self.device)
            labels_cat = torch.tensor([]).to(self.device)
    
            for client_nb, client_model in enumerate(selected_clients):
                labels_cat = torch.tensor([]).to(self.device)
                for data, label in self.trigger_loader:
                    data, label = data.to(self.device), label.to(self.device)
                    activation = client_model.model.get_activations_ablation(data)
                    clients_act[client_nb] = activation
                    labels_cat = label
                    break
            
            with torch.no_grad() : # Encountered memory leakage here otherwise. Is it because pytorch saves computation graphs for gm ????
                gm = compute_geometric_median(clients_act.cpu(), weights=None)
                #print(gm.termination)
                #if self.stored_gm : 
                #    gm.median = self.gm_memory*self.stored_gm.median + (1-self.gm_memory)*gm.median
    
            # Plot geomed and activations
            #self.plot_geomed(selected_clients, clients_act, gm.median)
            # self.plot_acts_gm_per_layer(selected_clients, clients_act, gm.median)
    
            # With CVAE
            clients_act = clients_act - gm.median.to(self.device)
            clients_act = torch.sigmoid(clients_act)
            for client_act in clients_act:
                condition = Utils.one_hot_encoding(labels_cat, self.num_classes, self.device).to(self.device)
                recon_batch, _, _ = self.cvae(client_act, condition)
                # recon_batch_bis, _, _ = self.cvae(client_act, condition)
                # print(f"Distance between two draws : {F.l1_loss(recon_batch, recon_batch_bis, reduction='mean').item()}")
                mse = F.mse_loss(recon_batch, client_act, reduction='mean').item()
                # redraw_err = abs(F.mse_loss(recon_batch_bis, client_act, reduction='mean').item() - mse)
                # clients_redraw_errors.append(redraw_err)
                # clients_energies.append(torch.mean(torch.square(client_act)).item())
                clients_re.append(mse)
            
            # benign_errors = []
            # attacker_errors = []
        
            # for re_val, client in zip(clients_re, selected_clients):
            #    if client.is_attacker:
                #    attacker_errors.append(re_val)
            #    else:
                #    benign_errors.append(re_val)
        
            #Combine and normalize errors (z-score)
            # all_errors = np.array(benign_errors + attacker_errors)
            # if len(all_errors) < 2:
            #    raise ValueError("Not enough data to plot.")
        
            # plt.figure(figsize=(12, 4.5))

            # sns.set_style("darkgrid")
        
            #Histogram (transparent, neutral gray)
            # plt.hist(benign_errors, bins=10, density=True, alpha=0.8, color='#377eb8', label='Benign', linewidth=0, edgecolor='black')
            # plt.hist(attacker_errors, bins=5, density=True, alpha=0.8, color='#b22222', label='Attacker', linewidth=0, edgecolor='black')

            # #Final styling
            # plt.xlabel("Reconstruction Error")
            # plt.ylabel("Count")
            # plt.title("Histogram of Client Reconstruction Errors")
            # plt.legend(loc='upper left')
            # plt.tight_layout()
            #plt.show()
            # plt.savefig("reconstruction_error_differences.pdf", dpi=300, bbox_inches='tight')

            # Compute new geomed from benign clients 
            # good_activations = clients_act[[not(client.is_suspect) for client in selected_clients]]
            # self.stored_gm = compute_geometric_median(good_activations.cpu(), weights=None)
            #self.stored_gm = gm
            
            # Plot activation representations 
            #for client_nb, client_act in enumerate(clients_act) : 
            #    condition = Utils.one_hot_encoding(labels_cat, self.num_classes, self.device).to(self.device)
            #    clients_lvs[client_nb] = self.cvae.get_latent_repr(client_act, condition)
            #self.plot_latent_vector(selected_clients, clients_lvs, title = "whole")
    
            # Plot highest and lowest re client reconstructed vectors
            # highest, lowest = np.argmax(clients_re), np.argmin(clients_re)
            # self.plot_reconstructed_vs_real(clients_act[highest], recon_batch[highest], clients_re[highest],
                                            # failed = True, benign = not(selected_clients[highest].is_attacker))
            # self.plot_reconstructed_vs_real(clients_act[lowest], recon_batch[lowest], clients_re[lowest],
                                            # failed = False, benign = not(selected_clients[lowest].is_attacker))
            
            
            # Real client identities (this is only used for visualisation, do not leak this into detection !!!!!!!!!!!)
            # attacker_indexes = [i for i in range(len(selected_clients)) if selected_clients[i].is_attacker]
            # benign_indexes = [i for i in range(len(selected_clients)) if not(selected_clients[i].is_attacker)]

            # Separate benign and attacker energies 
            # benigns_energy = [clients_energies[idx] for idx in benign_indexes]
            # attackers_energy = [clients_energies[idx] for idx in attacker_indexes]
            # benigns_re = [clients_re[idx] for idx in benign_indexes]
            # attackers_re = [clients_re[idx] for idx in attacker_indexes]
            # sorted_benigns_re = sorted(benigns_re)
            # sorted_attackers_re = sorted(attackers_re)
            # benigns_redraw_errors = [clients_redraw_errors[idx] for idx in benign_indexes]
            # attackers_redraw_errors = [clients_redraw_errors[idx] for idx in attacker_indexes]
            # sorted_benigns_redraw_errors = [err for _, err in sorted(zip(benigns_re, benigns_redraw_errors))]
            # sorted_attackers_redraw_errors = [err for _, err in sorted(zip(attackers_re, attackers_redraw_errors))]
            # benigns_re_uncertainty = [(gap/value)*100 for gap,value in zip(benigns_redraw_errors,benigns_re)]
            # attackers_re_uncertainty = [(gap/value)*100 for gap,value in zip(attackers_redraw_errors,attackers_re)]
            # benign_dist_over_uncertainty = [(sorted_benigns_re[idx+1]-sorted_benigns_re[idx])/redraw_err for idx,redraw_err in enumerate(sorted_benigns_redraw_errors[:-1])]
            # attacker_dist_over_uncertainty = [(sorted_attackers_re[idx+1]-sorted_attackers_re[idx])/redraw_err for idx,redraw_err in enumerate(sorted_attackers_redraw_errors[:-1])]

            # Examine metrics for Re propagation 
            # print(f"Re propagation stats : \
                #   \n {sorted_benigns_redraw_errors[0]}, {sorted_benigns_redraw_errors[-1]}, {sorted_attackers_redraw_errors[0]}, {sorted_attackers_redraw_errors[-1]} \
                #   \n {sorted_benigns_re[1]-sorted_benigns_re[0]}, {sorted_benigns_re[-1]-sorted_benigns_re[-2]}, \
# {sorted_attackers_re[1]-sorted_attackers_re[0]}, {sorted_attackers_re[-1]-sorted_attackers_re[-2]} \
                # \n {sorted_attackers_re[0] - sorted_benigns_re[-1]}")
            # print(f"Benigns uncertainty : {np.mean(benigns_re_uncertainty)}")
            # print(f"Attackers uncertainty : {np.mean(attackers_re_uncertainty)}") 
            # print(f"Benigns dist ratio : {np.mean(benign_dist_over_uncertainty)}")
            # print(benign_dist_over_uncertainty)
            # print(f"Attackers dist ratio : {np.mean(attacker_dist_over_uncertainty)}") 
            # print(attacker_dist_over_uncertainty)
    
            # Update detection boundaries 
            # self.attacker_lower_bounds.append(min(attackers_re))
            # self.attacker_upper_bounds.append(max(attackers_re))
            # self.benign_lower_bounds.append(min(benigns_re))
            # self.benign_upper_bounds.append(max(benigns_re))
            
            # Log activation and error energies
            # benign_odgs = [f"{signal:.5f}/{error:.5f}" for signal, error in zip(benigns_energy, benigns_re)]
            # attacker_odgs = [f"{signal:.5f}/{error:.5f}" for signal, error in zip(attackers_energy, attackers_re)]
            # print(f"Distance between attacker and benign : {F.l1_loss(recon_batch[lowest], recon_batch[highest], reduction='mean').item()}")
            # print(f"Signaux bénins : {benign_odgs}")
            # print(f"Signaux attaquants : {attacker_odgs}")
    
            return clients_re
    
    # Utils used by defense
    def compute_activations(self, selected_clients): 
        acts = torch.zeros(size=(len(selected_clients), self.activation_samples, self.activation_size)).to(self.device)
        with torch.no_grad():
            for client_nb, client_model in enumerate(selected_clients):
                for data, label in self.trigger_loader:
                    data, label = data.to(self.device), label.to(self.device)
                    activation = client_model.model.get_activations_ablation(data)
                    acts[client_nb] = activation
                    break
        
        return acts
        
    def train_cvae(self):
        if self.cvae_trained:
            print("CVAE is already trained, skipping re-training.")
            return
        
        # Storage
        reconstruction_errors = []
        sorted_reconstruction_errors = []
        redraw_errors = []
        sorted_redraw_errors = []
        dist_over_redraw_err = []

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
                activation = input_cvae_model.get_activations_ablation(data)
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
                activation = input_cvae_model.get_activations_ablation(data)
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

                # if epoch == num_epochs-1 :
                    # with torch.no_grad():
                        # recon_batch_bis, _, _ = self.cvae(activation, condition)
                    # redraw_err = abs(F.mse_loss(recon_batch_bis, activation, reduction='mean').item() - mse)
                    # redraw_errors.append(redraw_err.item())
                    # reconstruction_errors.append(mse.item())

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
        
        # Separate benign and attacker energies 
        #redraw_errors = redraw_errors.numpy()
        # print(reconstruction_errors)
        # print(redraw_errors)
        # sorted_reconstruction_errors = sorted(reconstruction_errors)
        # sorted_redraw_errors = [err for _, err in sorted(zip(reconstruction_errors, redraw_errors))]
        # dist_over_redraw_err = [(sorted_reconstruction_errors[idx+1]-sorted_reconstruction_errors[idx])/redraw_err for idx,redraw_err in enumerate(sorted_redraw_errors[:-1])]

        # print(f"cvae_dist_over_redraw_err_list : {dist_over_redraw_err}")

        # self.dist_over_redraw = np.mean(dist_over_redraw_err)
        # print(self.dist_over_redraw)

        #self.plot_cvae_loss(mse_over_epochs, KL_over_epochs, loss_over_epochs)
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
                activation = input_cvae_model.get_activations_ablation(data)
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
                activation = input_cvae_model.get_activations_ablation(data)
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
        #self.plot_cvae_loss(mse_over_epochs, KL_over_epochs, loss_over_epochs)
        self.cvae_trained = True
        
    def get_clusters(self, reconstruction_errors): 
        # hdb = HDBSCAN(min_cluster_size = 10)
        hdb = MeanShift(n_jobs=-1, max_iter=50)
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
        # Filtering
        highest_re_cluster_id = dbscan_result[highest_re_index]
        #self.plot_clusters(reconstruction_errors, dbscan_result, sorted(cluster_ceilings), highest_re_cluster_id)
        return (dbscan_result != highest_re_cluster_id)
        #return(reconstruction_errors.flatten() <= sorted(cluster_ceilings)[-2])
    
    def trust_propagation(self, selected_clients, clients_re):
        eps = self.lamda
        sorted_clients, sorted_re = [client for _, client in sorted(zip(clients_re, selected_clients))], sorted(clients_re)
        unsorted_pairs = [[client, re] for re, client in zip(clients_re, selected_clients)]
        sorted_pairs = sorted(unsorted_pairs, key = lambda x:x[1])
        # threshold = (sorted_re[-1] - sorted_re[0]) * eps
        threshold = (sorted_pairs[-1][1] - sorted_pairs[0][1]) * eps
        # good_updates = [sorted_clients[0]] 
        good_updates = [sorted_pairs[0][0]]
        # for idx,client in enumerate(sorted_clients[:-1]) :
        for idx,_ in enumerate(sorted_pairs[:-1]) :
            break_idx = idx+1
            #if (sorted_re[idx+1]-sorted_re[idx]) > threshold : # If next has much higher re, is probably an attacker
            if (sorted_pairs[idx+1][1]-sorted_pairs[idx][1]) > threshold :
                break 
            #good_updates.append(sorted_clients[idx+1])
            good_updates.append(sorted_pairs[idx+1][0])
        # Part for plotting
        fig, ax = plt.subplots()
        ax.set(title = f"Trust propagation")
        sns.stripplot(sorted_re[:break_idx], ax = ax, color ='b', orient = 'h')
        sns.stripplot(sorted_re[break_idx:], ax = ax, color = 'r', orient = 'h')
        ax.axvline(x=np.mean(sorted_re), linestyle = '--', color = 'g')
        plt.xlabel("Reconstruction error")
        if not os.path.exists(os.path.join(self.dir_path,"re_filtering")):
            os.makedirs(os.path.join(self.dir_path,"re_filtering"))
        plt.savefig(os.path.join(self.dir_path,"re_filtering",f"{datetime.datetime.now()}.png"))
        plt.close()
        return(good_updates)
    
    def get_benigns_re(reconstruction_errors, selected_clients): 
        return [re for re,client in zip(reconstruction_errors, selected_clients) if not(client.is_attacker)]
    
    def get_attacker_re(reconstruction_errors, selected_clients): 
        return [re for re,client in zip(reconstruction_errors, selected_clients) if client.is_attacker]

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
    
    def plot_latent_vector(self, clients, latent_vectors, title): 
        sns.set_theme()
        # fig, ax = plt.subplots()
        latent_vectors_df = pd.DataFrame(torch.mean(latent_vectors, dim=1).detach().cpu().numpy())
        # Compute PCA
        pca = PCA(n_components=2) 
        pca.fit(latent_vectors_df.T)
        dimred_latent_vectors = pca.components_.T
        # Plot 
        attacker_indexes = pd.Series([client.is_attacker for client in clients], index=latent_vectors_df.index)
        benign_indexes = pd.Series([not(client.is_attacker) for client in clients], index=latent_vectors_df.index)
        attacker_latent_vectors_df = dimred_latent_vectors[attacker_indexes]
        benign_latent_vectors_df = dimred_latent_vectors[benign_indexes]
        strip_plot = sns.scatterplot(benign_latent_vectors_df, color = 'b', legend = False) #, orient='h')
        sns.scatterplot(attacker_latent_vectors_df,ax = strip_plot, color = 'r', legend = False) #, orient='h')
        #plt.xlim(-0.6,0.6)
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
        color_map = "Blues" if (color == 'b') else "Reds"
        savepath = "failed_reconstruction" if failed else "reconstructed_successfully"
        # Layers split
        layersizes = [0,18432, 21632, 22208, 22464]
        layersides = [128,64,24,16]
        # layersizes = [0, 5184, 10368]
        layersizes = [0,5184]
        acts, reconstructed_acts = acts.detach().cpu().numpy().mean(axis=0), reconstructed_acts.detach().cpu().numpy()
        print(reconstructed_acts.shape)
        # Create figures
        fig, axs = plt.subplots(len(layersizes)-1,3, figsize =(50,50)) 
        fig.suptitle(f"reconstruction error : {reconstruction_error}")
        # Plot real acts 
        for idx in range(len(layersizes)-1): 
            #ax = axs[idx,0]
            ax = axs[0]
            ax.set_title(f"layer {idx}")
            # Vector
            #ax.plot(acts[layersizes[idx]:layersizes[idx+1]], color = color)
            # Heatmap 
            # activation_map = acts[layersizes[idx]:layersizes[idx+1]].reshape(layersides[idx],-1)
            activation_map = acts[layersizes[idx]:layersizes[idx+1]].reshape(72,-1)
            ax.imshow(activation_map, cmap = color_map)
        # Plot reconstructed acts 
        for idx in range(len(layersizes)-1): 
            #ax = axs[idx,1]
            ax = axs[2]
            ax.set_title(f"layer {idx} (reconstructed)")
            # Vector
            #ax.plot(reconstructed_acts[layersizes[idx]:layersizes[idx+1]], color = color)
            # Heatmap 
            # activation_map = reconstructed_acts[layersizes[idx]:layersizes[idx+1]].reshape(layersides[idx],-1)
            activation_map = reconstructed_acts[layersizes[idx]:layersizes[idx+1]].reshape(72,-1)
            ax.imshow(activation_map, cmap = color_map)
        # Plot diff   
        for idx in range(len(layersizes)-1): 
            #ax = axs[idx,1]
            ax = axs[1]
            ax.set_title(f"layer {idx} (reconstruction diff)")
            # Heatmap 
            orig_map = acts[layersizes[idx]:layersizes[idx+1]].reshape(72,-1)
            recon_map = reconstructed_acts[layersizes[idx]:layersizes[idx+1]].reshape(72,-1)
            diff = skimg.compare_images(orig_map, recon_map, method='diff')
            ax.imshow(activation_map, cmap = 'gray')
        # Save figure 
        plt.savefig(os.path.join(self.dir_path,savepath,f"{datetime.datetime.now()}.png"))
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
    
    def plot_decision_boundaries(self): 
        sns.set_theme()
        # Create figures
        fig, ax = plt.subplots(1,1, figsize =(20,20)) 
        ax.plot(self.benign_lower_bounds, color = 'b', label ='bening (lowest)')
        ax.plot(self.benign_upper_bounds, color = 'b', label ='benign (highest)')
        ax.plot(self.attacker_lower_bounds, color = 'r', label = "attacker (lowest)")
        ax.plot(self.attacker_upper_bounds, color = 'r', label ='attacker (highest)')
        ax.plot(self.decision_boundaries, color = 'orange', linestyle = '--', label ='threshhold (mean)')
        ax.plot(self.decision_boundaries_weighted, color = 'green', linestyle = '--', label ='threshhold (2 x mean)')
        ax.legend()
        plt.xlabel("FL rounds")
        plt.ylabel("Reconstruction error")
        ax.set(title = "Client reconstruction errors over rounds")
        # Save figure 
        plt.savefig(os.path.join(self.dir_path,"decision_boundaries.png"))
        #plt.show()
        plt.close()
    
    def plot_activation_maps(self, axs, activations, rows, color): 
        sns.set_theme()
        for ax, layer, L in zip(axs, activations, rows): 
            imdim = layer.shape[-1]
            image = np.zeros(((layer.size//L)*imdim,L*imdim))
            for channel in range(layer.shape[0]) :
                j = channel % L 
                i = channel // L
                image[(i*imdim):(i*imdim+1), j*imdim:(j+1)*imdim] = layer[channel,:,:]
                ax.imshow(image, cmap = color)

        # Save figure 
        plt.savefig(os.path.join(self.dir_path,"decision_boundaries.png"))
        #plt.show()
        plt.close()
        
    def plot_acts_over_time(self, acts_hist, layername): 
        pass 



