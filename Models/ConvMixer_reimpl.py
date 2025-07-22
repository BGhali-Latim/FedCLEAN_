import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import torch.nn as nn

class MixerBlock(nn.Module):
    def __init__(self, dim, depth, kernel_size = 3, patch_size = 7):
        super().__init__()
        self.depth_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                              nn.GELU(),)
                              #nn.BatchNorm2d(dim))
        self.pointwise_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1),
                              nn.GELU(),
                              nn.BatchNorm2d(dim))

    def forward(self, x):
        residual = self.depth_conv(x) + x 
        x = self.pointwise_conv(residual)
        return x
    
    def get_activation_without_residue_1(self, x): 
        with torch.no_grad(): 
            return self.depth_conv(x)
    
    def get_activation_without_residue_2(self, x): 
        with torch.no_grad(): 
            residual = self.depth_conv(x) + x 
            x = self.pointwise_conv(residual)
            return x


class ConvMixer(nn.Module):
    def __init__(self, dim, depth, num_channels =1, kernel_size = 3, patch_size = 7, n_classes = 10, device = 'cuda'):
        super(ConvMixer, self).__init__()
        # Network depth 
        self.depth = depth 
        # Network blocks
        # self.patch_embedding = nn.Sequential(nn.Conv2d(num_channels, dim, kernel_size=patch_size, stride=patch_size),
                                # nn.GELU(),
                                # nn.BatchNorm2d(dim))
        self.patch_embedding = nn.Conv2d(num_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding_act = nn.Sequential(nn.GELU(), 
                                                 nn.BatchNorm2d(dim))
        # self.mixer_layers = [MixerBlock(dim, depth, kernel_size, patch_size).to(device)] * self.depth
        self.mixer_layer_1 = MixerBlock(dim, depth, kernel_size, patch_size).to(device)
        self.classif_layer = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                nn.Flatten(),
                                nn.Linear(dim, n_classes))

    def forward(self, x): 
        x = x.view(-1,1,28,28)
        x = self.patch_embedding(x) 
        x = self.patch_embedding_act(x) 
        #print(f"patch size : {x.size()}")
        x = F.dropout(x, p=0.1, training=self.training)
        # for i in range(self.depth) :
            # x = self.mixer_layers[i](x)
            # x = F.dropout(x, p=0.1, training=self.training)
        x = self.mixer_layer_1(x)
        x = F.dropout(x, p=0.1, training=self.training)
        y = self.classif_layer(x) 
        return y
    
    def get_activations(self, x): 
        with torch.no_grad():
            x = x.view(-1,1,28,28)
            embeddings = self.patch_embedding(x) 
            x = self.patch_embedding_act(embeddings) 
            mixer_acts = self.mixer_layer_1.get_activation_without_residue_1(x)
        return torch.cat((torch.flatten(embeddings, start_dim=1), 
                          torch.flatten(mixer_acts, start_dim=1),
                          ),dim=1)

    