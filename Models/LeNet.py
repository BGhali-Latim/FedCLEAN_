# lenet.py

import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            
            nn.Linear(84, num_classes),
        )
        
    def forward(self, x):
        x = x.view(-1,1,28,28)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

