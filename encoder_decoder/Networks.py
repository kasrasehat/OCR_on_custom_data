import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, base_filters=64, adaptive_pool_size=(90, 90)):
        super(Encoder, self).__init__()

        # Initial layers before splitting into pathways
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, base_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.LeakyReLU()
        )

        # Define deeper pathways with multiple layers
        self.pathway1 = self._make_layers(base_filters, kernel_size=3, num_layers=5)
        self.pathway2 = self._make_layers(base_filters, kernel_size=7, num_layers=5)
        self.pathway3 = self._make_layers(base_filters, kernel_size=9, num_layers=5)

        # Reduction to 128 channels for each pathway
        self.reduce_channels = nn.Conv2d(base_filters * 3, 128, kernel_size=1, stride=1)

        # Additional convolutional layers after channel reduction
        self.additional_convs = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(adaptive_pool_size)
        )

        # Additional downsampling convolutional layer
        self.downsample = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, stride=3, padding=3),  # This will reduce 90x90 to 30x30
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )

        # Calculate the size of the feature map after downsampling
        downsampled_output_size = (30, 30)  # This is the expected output size after downsampling

        # Intermediate dense layer to extract more complex features
        self.intermediate_dense = nn.Linear(downsampled_output_size[0] * downsampled_output_size[1] * 128, 512)

        # Final feature vector layer to output a vector of size 116
        self.feature_vector_layer = nn.Linear(512, 116)

        # Weight initialization
        self.apply(self.initialize_weights)

    def _make_layers(self, in_channels, kernel_size, num_layers, stride=1):
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2))
            layers.append(nn.BatchNorm2d(in_channels))
            layers.append(nn.LeakyReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        # Apply the initial convolutional layers
        x = self.initial_conv(x)

        # Apply each deeper pathway
        p1 = self.pathway1(x)
        p2 = self.pathway2(x)
        p3 = self.pathway3(x)

        # Concatenate the outputs along the depth
        concatenated_features = torch.cat((p1, p2, p3), 1)

        # Reduce channel dimensions
        reduced_features = self.reduce_channels(concatenated_features)

        # Apply the additional convolutions
        features = self.additional_convs(reduced_features)

        # Downsample the feature maps to reduce the size
        features = self.downsample(features)

        # Flatten the features for the dense layers
        flattened_features = features.view(features.size(0), -1)

        # Apply the intermediate dense layer
        intermediate_features = self.intermediate_dense(flattened_features)

        # Get the final feature vector
        feature_vector = self.feature_vector_layer(intermediate_features)

        return features.float(), feature_vector.float()

    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            nn.init.constant
