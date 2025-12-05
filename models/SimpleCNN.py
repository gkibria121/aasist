import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, d_args):
        """
        Simple CNN for audio classification
        Works directly on raw audio waveforms
        
        Args:
            d_args: Dictionary containing model configuration
        """
        super(Model, self).__init__()
        
        self.d_args = d_args
        
        # Get config parameters with defaults
        num_classes = d_args.get("num_classes", 2)
        conv_channels = d_args.get("conv_channels", [32, 64, 128])
        kernel_size = d_args.get("kernel_size", 3)
        dropout = d_args.get("dropout", 0.5)
        
        # Simple 1D convolutions on raw audio
        self.conv1 = nn.Conv1d(1, conv_channels[0], kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(conv_channels[0])
        
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(conv_channels[1])
        
        self.conv3 = nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=kernel_size, padding=1)
        self.bn3 = nn.BatchNorm1d(conv_channels[2])
        
        # Max pooling
        self.pool = nn.MaxPool1d(kernel_size=4)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_channels[2], 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x, Freq_aug=False):
        # x is raw audio: (batch, samples)
        # Add channel dimension: (batch, 1, samples)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        # Store features
        last_hidden = x
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        
        return last_hidden, output


 