# custom_resnet_pre_activation.py

import torch
import torch.nn as nn
from torchvision import models

class CustomResNet(nn.Module):
    def __init__(self, pretrained=True, 
                layers_to_extract=['conv1','layer4.conv2'],
                layers_to_train=['layer2','layer4','fc']):
        super(CustomResNet, self).__init__()
        # Load the pretrained ResNet model
        self.resnet = models.resnet18(weights='DEFAULT')
        
        # Modify the final layer to match CIFAR-10 classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 10)

        #print(self.conv1)
        
        # Layers to extract
        self.layers_to_extract = layers_to_extract if layers_to_extract else []
        self.extracted_features = {}

        # Freeze pretrained layers 
        self.layers_to_train = layers_to_train
        self._set_trainable()


    def forward(self, x):
        x = x.view(-1,3,32,32)
        x = self.resnet(x)
        return x
    
    def _set_trainable(self): 
        #pass
        for p in self.resnet.parameters() : 
            p.requires_grad = False 
        for layer_name in self.layers_to_train : 
            layer = dict([*self.resnet.named_modules()])[layer_name]
            for param in layer.parameters(): 
                param.requires_grad = True
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.extracted_features[name] = output.detach()  # Extract input to activation
        return hook
    
    def _register_hooks(self):
        for layer_name in self.layers_to_extract:
            layer = dict([*self.resnet.named_modules()])[layer_name]
            layer.register_forward_hook(self.get_activation(layer_name))
    
    def get_activations(self, x): 
        self._register_hooks()
        self.extracted_features = {}  # Reset extracted features
        x = x.view(-1,3,32,32)
        x = self.resnet(x)
        return self.extracted_features
