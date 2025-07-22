import torch 
import torchvision.transforms.functional as TF

from tqdm import tqdm

from Samplers.IID_sampler import ClientSampler
from custom_datasets.Datasets import SyntheticLabeledDataset


class NaturalSampler(ClientSampler): 
    def __init__(self, cf) -> None:
        super().__init__(cf)

    def distribute_non_iid_data(self, dataset): 
        train_data = self.load_dataset(dataset)
        # get the key of each writer datasets
        writers = sorted(train_data.keys())
        # If separated by writer
        subdatasets = []
        print("generating subdatasets")
        for writer in tqdm(writers) :
            # get the images and labels of the first writer as numpy array
            images = train_data[writer]['images'][:]
            labels = train_data[writer]['labels'][:]
            print(images.shape)
            print(labels.shape)
            # transform the images and labels to torch tensor
            images_tensor = TF.to_tensor(images).view(-1,28,28) #HERE
            labels_tensor = torch.from_numpy(labels).view(-1).long()
            # Dataset 
            tmp = SyntheticLabeledDataset(images_tensor, labels_tensor)
            subdatasets.append(torch.utils.data.DataLoader(
                tmp,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=4,
                pin_memory=True,
            ))
        return subdatasets
