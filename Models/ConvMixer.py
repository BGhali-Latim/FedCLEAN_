import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


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
        # self.depth_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                                # nn.GELU(),
                                # nn.BatchNorm2d(dim))
        # self.pointwise_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1),
                                # nn.GELU(),
                                # nn.BatchNorm2d(dim))
        # self.conv_block = nn.Sequential(Residual(self.depth_conv), self.pointwise_conv)
        # self.mixer_blocks = [nn.Sequential(Residual(nn.Sequential(nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"), 
                                                                #   nn.GELU(), 
                                                                #   nn.BatchNorm2d(dim))),
                                            # nn.Conv2d(dim, dim, kernel_size=1), 
                                            # nn.GELU(), 
                                            # nn.BatchNorm2d(dim))] * depth
        #self.mixer_block_1 = nn.Sequential(Residual(nn.Sequential(nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"), 
        #                                                        nn.GELU(),)), 
        #                                                        # nn.BatchNorm2d(dim))),
        #                                  nn.Conv2d(dim, dim, kernel_size=1), 
        #                                  nn.GELU(),
        #                                  nn.BatchNorm2d(dim))
        self.mixer_block_1 = nn.Sequential(Residual(nn.Sequential(nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"), 
                                                                nn.GELU(),)), 
                                                                # nn.BatchNorm2d(dim))),
                                          nn.Conv2d(dim, dim, kernel_size=1)) 
        self.mixer_block_1_act = nn.Sequential(nn.GELU(),
                                            nn.BatchNorm2d(dim))
        self.classif_layer = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                nn.Flatten(),
                                nn.Linear(dim, n_classes))

    def forward(self, x): 
        x = x.view(-1,1,28,28)
        x = self.patch_embedding(x) 
        x = self.patch_embedding_act(x) 
        #print(f"patch size : {x.size()}")
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.mixer_block_1(x) 
        x = self.mixer_block_1_act(x) 
        x = F.dropout(x, p=0.1, training=self.training)
        #x = self.mixer_block_2(x) 
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = self.mixer_block_3(x) 
        y = self.classif_layer(x) 
        return y
    
    def get_activations(self, x): 
        x = x.view(-1,1,28,28)
        with torch.no_grad():
            x = x.view(-1,1,28,28)
            embeddings = self.patch_embedding(x) 
            x = self.patch_embedding_act(embeddings) 
            mixer_acts = self.mixer_block_1(x) 
        return torch.cat((torch.flatten(embeddings, start_dim=1), 
                          torch.flatten(mixer_acts, start_dim=1),
                          ),dim=1)

    