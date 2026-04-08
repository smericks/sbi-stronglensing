import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.models import resnet34, ResNet34_Weights

from sbi.neural_nets.embedding_nets import CNNEmbedding


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, output_dim=64, weights=None):
        super(ResNetFeatureExtractor, self).__init__()
        
        # Load ResNet34 architecture
        # num_classes controls the number of features in the FC layer
        self.resnet = resnet34(weights=weights,num_classes=output_dim) # TODO: this arg?

        # NOTE: hardcoded to 1-channel input!! 
        #   (the hardcoding of 64, 7, 2, 3 matches the original implementation)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        
    def forward(self, x):
        # Ensure input is float
        x = x.float()
            
        # Add channel dimension if missing
        if x.dim() == 3:  # [batch, height, width]
            x = x.unsqueeze(1)  # -> [batch, channel, height, width]
            
        # Extract features through ResNet
        x = self.resnet.forward(x)
        
        return x

def get_feature_extractor(network_architecture="cnn", output_dim=64, weights=None, input_shape=(180, 180), 
                          in_channels=1, out_channels_per_layer=[32, 64, 128, 256], num_conv_layers=4, num_linear_layers=2, num_linear_units=256, kernel_size=5, pool_kernel_size=2):
    if network_architecture == "resnet34":
        model = ResNetFeatureExtractor(output_dim=output_dim, weights=weights)
    elif network_architecture == "cnn":
        model = CNNEmbedding(
            input_shape= tuple(input_shape) if isinstance(input_shape, list) else input_shape,
            in_channels=in_channels,
            out_channels_per_layer=out_channels_per_layer,
            num_conv_layers=num_conv_layers,
            num_linear_layers=num_linear_layers,
            num_linear_units=num_linear_units,
            output_dim=output_dim,
            kernel_size=kernel_size,
            pool_kernel_size=pool_kernel_size
        )
    else:
        raise ValueError(f"Unsupported network architecture: {network_architecture}")

    return model