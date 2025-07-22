import os
import json
import numpy as np
import random
import torch
from math import floor
from statistics import mean
import pickle

from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets
from torchvision.transforms import transforms
import torch.nn.functional as F
import gc

import torchvision.transforms.functional as TF

from Client.Client import Client
from custom_datasets.Datasets import SyntheticLabeledDataset
from custom_datasets import FEMNIST_pytorch 

class Utils:
    def __init__(self):
        pass

    @staticmethod
    def distribute_iid_data_among_clients(num_clients, batch_size, dataset):
        if dataset == "MNIST" :
            data = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
            print('ok')
        elif dataset == "FashionMNIST" :
            data = datasets.FashionMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        elif dataset == "CIFAR10" : 
            trans_cifar_train = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Normalize(
                    (0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
            data = datasets.CIFAR10(root='./data', train=True, transform=trans_cifar_train, download=True)
        data_size = len(data) // num_clients
        return [
            DataLoader(Subset(data, range(i * data_size, (i + 1) * data_size)), batch_size=batch_size, shuffle=True, drop_last=False)
            for i in range(num_clients)]
    
    @staticmethod
    def distribute_non_iid_data_among_clients(num_clients, batch_size, dataset):

        size_def_dict = { 'MNIST' : {100 : 9400, 1000 : 2000,},
                        "FashionMNIST" : {100 : 9400, 1000 : 2000,},
                        "CIFAR10" : {1000 : 1040},
                        }

        if dataset == "MNIST" :
            trans_mnist_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])
            train_data = datasets.MNIST(root='./data', train=True, transform=trans_mnist_train                                        
                , download=True)

        elif dataset == "FashionMNIST" :
            trans_fashionmnist_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,), (0.3530,))
                    ])
            train_data = datasets.FashionMNIST(root='./data', train=True, transform=trans_fashionmnist_train, download=True)

        elif dataset == "CIFAR10" : 
            trans_cifar_train = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Normalize(
                    (0.5,0.5,0.5), (0.5,0.5,0.5)),])    
            train_data = datasets.CIFAR10(root='./data', train=True, transform=trans_cifar_train, download=True)
            train_data.targets = torch.tensor(train_data.targets)
            train_data.data = torch.tensor(train_data.data)

        elif dataset == "FEMNIST": 
            import h5py
            #train_data = femnist.FEMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
            #data_size = len(train_data) // num_clients
            #return [
            #    DataLoader(Subset(train_data, range(i * data_size, (i + 1) * data_size)),
            #               batch_size=batch_size, shuffle=True, drop_last=False)
            #    for i in range(num_clients)]
            # load the dataset
            print("loading dataset")
            ds = h5py.File('./data/write_digits.hdf5', 'r')
            # get the key of each writer datasets
            writers = sorted(ds.keys())
            subdatasets = []
            print("generating subdatasets")
            for writer in tqdm(writers) :
                # get the images and labels of the first writer as numpy array
                images = ds[writer]['images'][:]
                labels = ds[writer]['labels'][:]
                # transform the images and labels to torch tensor
                images_tensor = TF.to_tensor(images).view(-1,28,28) #HERE
                labels_tensor = torch.from_numpy(labels).view(-1).long()
                # Dataset 
                tmp = SyntheticLabeledDataset(images_tensor, labels_tensor)
                subdatasets.append(torch.utils.data.DataLoader(
                    tmp,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=4,
                    pin_memory=True,
                ))
            return subdatasets

        indices = [torch.where(train_data.targets == idx)[0] for idx in range(0, 10)]
        #print(len(indices))
        #print([len(elem) for elem in indices])

        subdatasets = []
        backdoor_sets = []
        tuples_set = []

        size_def = size_def_dict[dataset][num_clients]

        for k in tqdm(range(1,num_clients+1)):   
            sample_size = int(floor(size_def/(k+5)))+20

            while True :
                i, j = random.sample(range(0, 10), k=2)
                if (len(indices[i])+len(indices[j]))>=2*sample_size : 
                    break

            tuples_set.append([i, j])

            if len(indices[i])<sample_size :
                indice_i = np.random.choice(range(len(indices[i])), size=len(indices[i]), replace=False)
                indice_j = np.random.choice(range(len(indices[j])), size=2*sample_size-len(indices[i]), replace=False)
                #class_counts[i]+=len(indices[i])
                #class_counts[j]+=2*size-len(indices[i])
            elif len(indices[j])<sample_size :
                indice_i = np.random.choice(range(len(indices[i])), size=2*sample_size-len(indices[j]), replace=False)
                indice_j = np.random.choice(range(len(indices[j])), size=len(indices[j]), replace=False)
                #class_counts[i]+=2*size-len(indices[j])
                #class_counts[j]+=len(indices[j])
            else : 
                indice_i = np.random.choice(range(len(indices[i])), size=sample_size, replace=False)
                indice_j = np.random.choice(range(len(indices[j])), size=sample_size, replace=False)
                #class_counts[i]+=size
                #class_counts[j]+=size

            selected_i = indices[i][indice_i]
            selected_j = indices[j][indice_j]

            combined = torch.cat((selected_i, indices[i]))
            uniques, counts = combined.unique(return_counts=True)
            indices[i] = uniques[counts == 1]

            combined = torch.cat((selected_j, indices[j]))
            uniques, counts = combined.unique(return_counts=True)
            indices[j] = uniques[counts == 1]

            selected = torch.cat((selected_i, selected_j))

            data_selected = train_data.data[selected]
            label_selected = train_data.targets[selected]

            #print(data_selected)
            #print(data_selected.size())

            tmp = torch.utils.data.TensorDataset(((data_selected.float() / 255.) - 0.1307) / 0.3081, label_selected)

            subdatasets.append(torch.utils.data.DataLoader(
                tmp,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=4,
                pin_memory=True,
            ))
        print(sum([len(loader.dataset) for loader in subdatasets]))
        # Utils.plot_hist([len(loader.dataset) for loader in subdatasets], bins=50, title_info="Data distribution in non-iid setting", x_info="client dataset size", y_info="number of clients", save_path="./hist.pdf")
        print([len(loader.dataset) for loader in subdatasets])
        return subdatasets
    
    @staticmethod
    def plot_data_dist(clients, savepath = ""):
        sizes = sorted([len(client.dataloader.dataset) for client in clients])
        sns.set_theme()
        fig, ax = plt.subplots()
        ax.hist(sizes[:-1], sizes)     
        ax.set(xlabel="Number of samples",
               ylabel="Number of clients", title="Client data distribution for MNIST")
        ax.text(500, 200, f"total samples = {sum(sizes)}")
        plt.show()

    @staticmethod
    def gen_database(num_clients, batch_size, dataset):
        return Utils.distribute_iid_data_among_clients(num_clients, batch_size, dataset)

    @staticmethod
    def gen_clients(num_clients, attacker_ratio, attack_type, train_data, backdoor_label = '0'):
        total_clients = num_clients
        num_attackers = int(total_clients * attacker_ratio)

        if attack_type == "MajorityBackdoor" : # Clients are already ordered by data size in non-IID
            attacker_flags = [True] * num_attackers + [False] * (total_clients - num_attackers)
        elif attack_type == "TargetedBackdoor" : 
            attacker_flags = []
            for i in tqdm(range(total_clients)):
                for data,labels in train_data[i] :
                    if (backdoor_label in labels) : 
                        attacker_flags.append(True)
                    else : 
                        attacker_flags.append(False) 
        else :
            attacker_flags = [True] * num_attackers + [False] * (total_clients - num_attackers)
            np.random.shuffle(attacker_flags)
        
        clients = [Client(ids=i, dataloader=train_data[i], is_attacker=attacker_flags[i], attack_type=attack_type)
        for i in tqdm(range(total_clients))] 

        # For backdoor clients, check how many attackers have been eliminated due to not having the backdoor class 
        real_num_attackers = np.sum([client.is_attacker for client in clients])
        print(f"Final number of attackers : {real_num_attackers} / {num_attackers} ({(real_num_attackers/num_attackers)*attacker_ratio*100:.2f}%)")

        return clients

    @staticmethod
    def cvae_loss(recon_x, x, mu, logvar):
        mse = F.mse_loss(recon_x, x, reduction='mean')
        # MSE = F.binary_cross_entropy(recon_x, x, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # print(f"cvae loss : mse : {mse} kld : {kld}")
        return mse, kld

    @staticmethod
    def plot_accuracy(accuracy, x_info="round", y_info="Test Accuracy", title_info= "provide a title", save_path=None):
        sns.set_theme()
        fig, ax = plt.subplots()
        sns.set(rc={'figure.figsize':(11,9)})
        ax.plot(range(1, len(accuracy) + 1), accuracy)
        ax.set_ylim([0,1.1])
        ax.set_xlabel(x_info)
        ax.set_ylabel(y_info)
        ax.set_title(title_info)
        ax.text(60, 0.1, f"best test accuracy : {max(accuracy)*100:.2f}%\
         \nAchieved on round {np.argmax(np.array(accuracy))+1}")
        plt.savefig(save_path)
        #plt.show()

    @staticmethod
    def plot_hist(data, x_info="Values", y_info="Frequencies", title_info= "provide a title", bins=1000, save_path=None):
        sns.set_theme()
        fig, ax = plt.subplots()
        sns.set(rc={'figure.figsize':(11,9)})
        ax.set_title(title_info)
        ax.set_xlabel(x_info)
        ax.set_ylabel(y_info)
        ax.hist(data, bins=bins)
        plt.savefig(save_path)
        #plt.show()
    
    @staticmethod
    def plot_recall(attacker_info = [], benign_info = [], rounds = None, title_info = None, save_path = None): 
        sns.set_theme()        
        sns.set(rc={'figure.figsize':(11,9)})
        fig, ax = plt.subplots()
        ax.plot(range(rounds), attacker_info, color = 'black', label = "Attackers blocked")
        ax.plot(range(rounds), benign_info, color = 'r', label = "Benign clients passed")
        ax.legend()
        ax.set(title=title_info, xlabel="Rounds")
        ax.set_ylim([0,1.1])
        ax.set_xlim([0,rounds])
        ax.axhline(y = 0.5, linestyle = '--', color='black')
        ax.text(-10, 0.5, "y = 0.5")
        #ax.text(60, 0.25, f"{mean(precision_info)}")
        plt.savefig(save_path)

    @staticmethod
    def plot_histogram(hp, nb_attackers_passed_defence_history, nb_attackers_history,
                       nb_benign_passed_defence_history, nb_benign_history, config_fl,
                       attack_type, defence, dir_path, success_rate, attacker_ratio):
        rounds = np.arange(1, hp["nb_rounds"] + 1)

        height_attackers_passed_defense = np.array(nb_attackers_passed_defence_history)
        height_remaining_attackers = np.array(nb_attackers_history) - height_attackers_passed_defense

        height_benign_passed_defense = np.array(nb_benign_passed_defence_history)
        height_remaining_benign = np.array(nb_benign_history) - height_benign_passed_defense

        fig, ax = plt.subplots()

        ax.bar(rounds, height_attackers_passed_defense, color='red', edgecolor='black', alpha=0.5,
                label='Attackers Passed Defence')
        ax.bar(rounds, height_remaining_attackers, bottom=height_attackers_passed_defense, color='yellow',
                edgecolor='black', alpha=0.6, label='Total Attackers')

        ax.bar(rounds, height_benign_passed_defense, bottom=height_attackers_passed_defense + height_remaining_attackers, color='blue', edgecolor='black', alpha=0.5,
                label='Benign Clients Passed Defence')

        ax.bar(rounds, height_remaining_benign, bottom=height_benign_passed_defense + height_attackers_passed_defense + height_remaining_attackers, color='black',
                edgecolor='black', alpha=0.6, label='Total Benign Clients')

        ax.set_xlabel('Number of Rounds')
        ax.set_ylabel('Total Nb of Clients')
        ax.set_ylim(0, config_fl["nb_clients_per_round"])
        ax.set_title(f"Histogram for {attacker_ratio * 100}% of {attack_type} "
                  f"with {'Defence' if defence else 'No Defence'}")

        ax.legend()
        ax.text(1,45,f"Blocked {success_rate} of attacks", color = 'w', weight = "bold")
        plt.savefig(f"{dir_path}/{attack_type}_{'With defence' if defence else 'No defence'}_Histogram_{hp['nb_rounds']}.png")

        #plt.show()

    @staticmethod
    def test(model, device, loader):
        model.to(device).eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, labels in loader:
                data, labels = data.to(device), labels.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy
    
    def get_class_accuracies(model, device, class_datasets, nb_classes):
        model.to(device).eval()
        class_accuracies = {}

        for target_class in range(nb_classes) :
            correct, total = 0, 0
            target_loader = DataLoader(class_datasets[target_class], shuffle=False,drop_last=False)
            with torch.no_grad():
                for data, labels in target_loader:    
                    data, labels = data.to(device), labels.to(device)
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = correct / total
            class_accuracies[target_class] = accuracy
        return class_accuracies
    
    @staticmethod
    def get_loss(model, device, test_loader, criterion):
        model.to(device).eval()
        loss = 0
        with torch.no_grad():
            for data,labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                loss += criterion(predicted.float(), labels.float())
        return loss/len(test_loader)

    @staticmethod
    def test_backdoor(global_model, device, test_loader, attack_type, source, target, square_size):
        global_model.to(device).eval()
        total_source_labels, misclassified_as_target = 0, 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)

                tmp_data = data[labels==source].clone()
                tmp_labels = labels[labels==source].clone()

                if len(tmp_data) != 0 :
                    if attack_type == 'SquareBackdoor':
                        tmp_data[0, :square_size, :square_size] = 1.0

                    outputs = global_model(tmp_data)
                    _, predicted = torch.max(outputs, 1)

                    total_source_labels += tmp_labels.size(0)
                    misclassified_as_target += (predicted == target).sum().item()

        effectiveness = misclassified_as_target / total_source_labels if total_source_labels > 0 else 0
        return effectiveness
    
    @staticmethod
    def test_sourceless_backdoor(global_model, device, test_loader, attack_type, target, square_size):
        global_model.to(device).eval()
        total_labels, misclassified_as_target = 0, 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)

                tmp_data = data.clone()
                tmp_labels = labels.clone()

                if len(tmp_data) != 0 :
                    tmp_data[:, :square_size, :square_size] = 1.0

                    outputs = global_model(tmp_data)
                    _, predicted = torch.max(outputs, 1)
                    
                    total_labels += tmp_labels.size(0)
                    misclassified_as_target += (predicted == target).sum().item()
        effectiveness = misclassified_as_target / total_labels if total_labels > 0 else 0
        return effectiveness

    @staticmethod
    def get_test_data(size_trigger, dataset):
        if dataset == "MNIST" :
            trans_mnist_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
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
        elif dataset == "FEMNIST":
            data_test = FEMNIST_pytorch.FEMNIST(root='./data', 
                                        train=False, 
                                        transform=transforms.ToTensor(), 
                                        download=True)
        elif dataset == "CIFAR10" :
            trans_cifar_test = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                (0.2023, 0.1994, 0.2010)),])
            data_test = datasets.CIFAR10(root='./data',
                                        train=False,
                                        transform=trans_cifar_test,
                                        download=True)
        size_test = len(data_test) - size_trigger
        trigger_set, validation_set = random_split(data_test, [size_trigger, size_test])
        # Create data loaders
        trigger_loader = DataLoader(trigger_set, batch_size=size_trigger, shuffle=False, drop_last=False) if size_trigger else None
        test_loader = DataLoader(validation_set, batch_size=size_test, shuffle=False, drop_last=False)
        return trigger_loader, test_loader
    
    @staticmethod
    def split_train(train_loaders, size_train, size_trigger, size_test):
        random.shuffle(train_loaders)
        print(len(train_loaders))
        train_data = train_loaders[:size_train]
        print("preparing trigger")
        trigger_data = Utils.concat_loaders(train_loaders[size_train:size_train+size_trigger], size_trigger)
        print("preparing train")
        test_data = Utils.concat_loaders(train_loaders[size_train+size_trigger:], size_test)
        return train_data, trigger_data, test_data
    
    @staticmethod
    def concat_loaders(loader_list, batch_size):
        data, labels = [], []
        for loader in tqdm(loader_list) : 
            for data_batch, label_batch in loader :
                data.append(data_batch)
                labels.append(label_batch)
        data_tensor, label_tensor = torch.cat(data), torch.cat(labels)
        tmp = SyntheticLabeledDataset(data_tensor, label_tensor.view(-1))
        combined_loader =torch.utils.data.DataLoader(
            tmp,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )
        return combined_loader

    @staticmethod
    def select_clients(clients, nb_clients_per_round):
        selected_clients = random.sample(clients, nb_clients_per_round)
        return selected_clients

    @staticmethod
    def save_to_json(accuracies, dir_path, file_name):
        file_name = f"{dir_path}/{file_name}.json"
        with open(file_name, "w") as f:
            json.dump(accuracies, f)

    @staticmethod
    def read_from_json(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
    @staticmethod
    def save_latex_ready_metrics(metrics, dir_path, file_name): 
        metrics_string = f"FN/FP/ACC : {' / '.join(metrics)}"
        file_name = f"{dir_path}/{file_name}.json"
        with open(file_name, "w") as f:
            json.dump(metrics_string, f)

    @staticmethod
    def aggregate_models(clients):
        aggregated_state_dict = {}
        total_samples = sum([client.num_samples for client in clients])

        # Initialize
        for name, param in clients[0].model.state_dict().items():
            aggregated_state_dict[name] = torch.zeros_like(param).float()
        # Aggregate the clients' models
        for client in clients:
            num_samples = client.num_samples
            weight_factor = num_samples / total_samples
            client_state_dict = client.get_model().state_dict()

            for name, param in client_state_dict.items():
                aggregated_state_dict[name] += weight_factor * param
        return aggregated_state_dict

    @staticmethod
    def one_hot_encoding(label, num_classes, device):
        label = label.to(device)
        one_hot = torch.eye(num_classes).to(device)[label]
        return one_hot.squeeze(1).to(device)
    
    @staticmethod 
    def is_backdoor(attack_name): 
        backdoors_list = ["NaiveBackdoor", "SquareBackdoor", "NoiseBackdoor", "MajorityBackdoor", 
         "TargetedBackdoor", "SourcelessBackdoor", "DistBackdoor", "AlternatedBackdoor", "Neurotoxin"]
        return attack_name in backdoors_list
    
    @staticmethod
    def check_gpu_usage():
    # prints currently alive Tensors and Variables
        print(f"{len(gc.get_objects())} tensors loaded in memory")
        print(torch.cuda.mem_get_info())
        # print(torch.cuda.memory_summary())
            # try:
                # if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    # print(type(obj), obj.size())
            # except:
                # pass

    # ****** Functions related to FedCVAE

    @staticmethod
    def get_prod_size(model):
        size = 0
        for param in model.parameters():
            size += np.prod(param.weight.shape)
        return size
    
    # ****** Functions related to FedGuard
    @staticmethod
    def sample_from_normal(nb_samples, dim_samples, device):
        return torch.normal(mean=0.0, std=1.0, size=(nb_samples,dim_samples), device=device)
    
    @staticmethod
    def sample_from_cat(nb_samples, device):
        return torch.randint(low=0, high=10, size=(nb_samples,1), device=device)
    
    @staticmethod
    def create_heatmap(clients, nb_classes = 10, title = None, file_path = None): 
        # Create a figure with multiple subplots
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # make data
        data = np.zeros((nb_classes,len(clients)))
        for idx, client in enumerate(clients): 
            data[idx,:] = client.yield_heatmap_row()
        # make heatmap
        sns.heatmap(data,ax = ax, square = True)
        if title :
            ax.set_title(title)
        # Save heatmap
        plt.savefig(file_path)
    
    @staticmethod
    def plot_distrib(clients, nb_classes = 10, title = None, file_path = None):
        category_names = [str(i) for i in range(nb_classes)]              
        results = {} 
        indexes = list(range(len(clients))) 
        for i in range(len(clients)):
            #results[f"client {i+1}"] = clients[i].yield_heatmap_row()
            results[f"{i+1}"] = clients[i].yield_heatmap_row()
        labels = list(results.keys())
        data = np.array(list(results.values()))
        data_cum = data.cumsum(axis=1)
        category_colors = plt.colormaps['twilight'](
            np.linspace(0.15, 0.85, data.shape[1]))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.invert_yaxis()
        ax.xaxis.set_visible(True)
        ax.set_xlim(0, np.sum(data, axis=1).max())
        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            rects = ax.barh(labels, widths, left=starts, height=0.7,
                            label=colname, color=color)
            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            #ax.bar_label(rects, label_type='center', color=text_color)
        #ax.legend(ncol=2, bbox_to_anchor=(0, 1),
        #          loc='best', fontsize='small')
        if title :
            ax.set_title(title)
        ax.set(xlabel="")
        plt.savefig(file_path)
    
    @staticmethod
    def compare_distribs(clients, nb_classes = 10, file_path = None, storage = None, load = False):
        if load : 
            with open(storage, "rb") as f : 
                clients_bis = pickle.load(f) 
        else : 
            with open(storage, "wb") as f : 
                pickle.dump(clients, f)
        with open("base", "rb") as f : 
            clients_base = pickle.load(f) 
        category_names = [str(i) for i in range(nb_classes)]              
        results = {} 
        results_bis = {} 
        results_base = {} 
        indexes = list(range(len(clients))) 
        clients = sorted(clients, key= lambda client : client.num_samples, reverse=True)
        clients_bis = sorted(clients_bis, key= lambda client : client.num_samples, reverse = True)
        clients_base = sorted(clients_base, key= lambda client : client.num_samples, reverse = True)
        for i in range(len(clients)):
            #results[f"client {i+1}"] = clients[i].yield_heatmap_row()
            results[f"{i+1}"] = clients[i].yield_heatmap_row()
            results_bis[f"{i+1}"] = clients_bis[i].yield_heatmap_row()
            results_base[f"{i+1}"] = clients_base[i].yield_heatmap_row()
        labels = list(results.keys())
        data = np.array(list(results.values()))
        data_bis = np.array(list(results_bis.values()))
        data_base = np.array(list(results_base.values()))
        data_cum = data.cumsum(axis=1)
        data_cum_bis = data_bis.cumsum(axis=1)
        data_cum_base = data_base.cumsum(axis=1)
        category_colors = plt.colormaps['twilight_shifted'](
            np.linspace(0.15, 0.85, data.shape[1]))
        fig, (ax3,ax1,ax2) = plt.subplots(ncols=3, figsize=(11, 4))
        #ax3 = plt.subplot2grid( (4,4), [0,1], 2, 2 )
        #ax1 = plt.subplot2grid( (4,4), [2,0], 2, 2 )
        #ax2 = plt.subplot2grid( (4,4), [2,2], 2, 2 )
        ax1.invert_yaxis()
        ax1.xaxis.set_visible(True)
        ax1.set_xlim(0, np.sum(data, axis=1).max())
        ax2.invert_yaxis()
        ax2.xaxis.set_visible(True)
        ax2.set_xlim(0, np.sum(data_bis, axis=1).max())
        ax3.invert_yaxis()
        ax3.xaxis.set_visible(True)
        ax3.set_xlim(0, np.sum(data_base, axis=1).max())
        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            rects = ax1.barh(labels, widths, left=starts, height=0.7,
                            label=colname, color=color)
            widths_bis = data_bis[:, i]
            starts_bis = data_cum_bis[:, i] - widths_bis
            rects = ax2.barh(labels, widths_bis, left=starts_bis, height=0.7,
                            label=colname, color=color)
            widths_base = data_base[:, i]
            starts_base = data_cum_base[:, i] - widths_base
            rects = ax3.barh(labels, widths_base, left=starts_base, height=0.7,
                            label=colname, color=color)
        #ax.legend(ncol=2, bbox_to_anchor=(0, 1),
        #          loc='best', fontsize='small')
        fig.suptitle("Sample repartition of the MNIST classes across 20 clients")
        fig.supylabel("Clients") 
        fig.supxlabel("Number of samples") 
        ax3.set_title("IID")
        ax1.set_title("Dirichlet")
        ax2.set_title("Custom")
        fig.tight_layout()
        plt.savefig(file_path, bbox_inches='tight', dpi=300)
        



