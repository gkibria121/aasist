import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, d_args):
        """
        Simple CNN for audio classification
        
        Args:
            d_args: Dictionary containing model configuration
        """
        super(Model, self).__init__()
        
        self.d_args = d_args
        
        # Get config parameters with defaults
        input_channels = d_args.get("input_channels", 1)
        num_classes = d_args.get("num_classes", 2)
        conv_channels = d_args.get("conv_channels", [32, 64, 128])
        kernel_size = d_args.get("kernel_size", 3)
        pool_size = d_args.get("pool_size", 2)
        fc_hidden_dim = d_args.get("fc_hidden_dim", 256)
        dropout = d_args.get("dropout", 0.5)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, conv_channels[0], 
                               kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channels[0])
        
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], 
                               kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channels[1])
        
        self.conv3 = nn.Conv2d(conv_channels[1], conv_channels[2], 
                               kernel_size=kernel_size, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_channels[2])
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_channels[2] * 4 * 4, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, num_classes)
        
    def forward(self, x, Freq_aug=False):
        # Add channel dimension if needed
        if x.dim() == 3:
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
        
        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Store features for compatibility
        last_hidden = x
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        
        return last_hidden, output

 