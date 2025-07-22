import torch
import numpy as np
import random 

from collections import defaultdict

from Samplers.IID_sampler import ClientSampler

class DirichletSampler(ClientSampler): 
    def __init__(self, cf, method, alpha = 1) -> None:
        super().__init__(cf)
        self.method = method 
        self.alpha = alpha
        self.name = "Dirichlet"

    def distribute_non_iid_data(self, dataset): 
        train_data = self.load_dataset(dataset)
        if self.method == "Dirichlet_per_class" : 
            subdataset_indexes = self.distribute_per_class(train_data)
        elif self.method == "Dirichlet_per_client" :
            subdataset_indexes = self.distribute_per_client(train_data)
        subdatasets = []
        for client in range(self.num_clients) :
            selected = subdataset_indexes[client]
            data_selected = torch.tensor(train_data.data)[selected]
            label_selected = torch.tensor(train_data.targets)[selected]
            tmp = torch.utils.data.TensorDataset(data_selected.float(), label_selected)
            #tmp = self.perform_augmentation(tmp)
            subdatasets.append(torch.utils.data.DataLoader(
                tmp,
                batch_size=self.batch_size,
                shuffle=True,
                #drop_last=False,
                num_workers=4,
                pin_memory=True,
                drop_last = True #TODO
            ))
        print(f"Distributed {sum([len(loader.dataset) for loader in subdatasets])} samples among clients")
        print([len(loader.dataset) for loader in subdatasets])
        return subdatasets
    
    def distribute_per_class(self, train_data): 
        classes = {}
        # Get data indexes per label
        for ind, x in enumerate(train_data):
            _, label = x
            if label in classes:
                classes[label].append(ind)
            else:
                classes[label] = [ind]
        # Get real data size
        class_size = len(classes[0])
        no_classes = len(classes.keys())
        # Fill client sample index list
        per_participant_list = defaultdict(list)
        for n in range(no_classes):
            random.shuffle(classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(self.num_clients * [self.alpha]))
            for user in range(self.num_clients):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = classes[n][
                               :min(len(classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                classes[n] = classes[n][
                                   min(len(classes[n]), no_imgs):]
        return per_participant_list

    
