import torch 
import numpy as np
import random 
from math import floor

from tqdm import tqdm

from Samplers.IID_sampler import ClientSampler


class CAMSampler(ClientSampler): 
    def __init__(self, cf) -> None:
        super().__init__(cf)
        self.size_def_dict = { 'MNIST' : {10 : 25500, 100 : 9400, 20 : 15000, 1000 : 2000,},
                        "FashionMNIST" : {10 : 25500, 20 : 15000, 50 : 12500, 100 : 9400, 1000 : 2000,},
                        "FEMNIST" : {100 : 57000, 1000 : 1040},
                        "CIFAR10" : {20 : 15000, 100 : 7500, 1000 : 1040},
                        }
        self.name = "CAM"

    def distribute_non_iid_data(self, dataset): 
        train_data = self.load_dataset(dataset)
        size_def = self.size_def_dict[dataset][self.num_clients]
        indices = [torch.where(train_data.targets == idx)[0] for idx in range(0, 10)]
        subdatasets = []
        tuples_set = []
        for k in tqdm(range(1,self.num_clients+1)):   
            sample_size = int(floor(size_def/(k+5)))+20

            while True :
                i, j = random.sample(range(0, 10), k=2)
                if (len(indices[i])+len(indices[j]))>=2*sample_size : 
                    break
            tuples_set.append([i, j])

            if len(indices[i])<sample_size :
                indice_i = np.random.choice(range(len(indices[i])), size=len(indices[i]), replace=False)
                indice_j = np.random.choice(range(len(indices[j])), size=2*sample_size-len(indices[i]), replace=False)
            elif len(indices[j])<sample_size :
                indice_i = np.random.choice(range(len(indices[i])), size=2*sample_size-len(indices[j]), replace=False)
                indice_j = np.random.choice(range(len(indices[j])), size=len(indices[j]), replace=False)
            else : 
                indice_i = np.random.choice(range(len(indices[i])), size=sample_size, replace=False)
                indice_j = np.random.choice(range(len(indices[j])), size=sample_size, replace=False)

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

            tmp = torch.utils.data.TensorDataset(data_selected.float(), label_selected)
            #tmp = self.perform_augmentation(tmp)

            subdatasets.append(torch.utils.data.DataLoader(
                tmp,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=4,
                pin_memory=True,
            ))
        print(f"Distributed {sum([len(loader.dataset) for loader in subdatasets])} samples among clients")
        print([len(loader.dataset) for loader in subdatasets])
        return subdatasets
