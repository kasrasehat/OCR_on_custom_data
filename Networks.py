
import torch.utils.data
import torchvision
from torchvision import transforms
import cv2
import torchvision.models as models
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np

# model_name = 'resnet152'
# checkpoint = 'E:/codes_py/Larkimas/checkpoints/resnet152_10_0.9719.pth'
# model = torchvision.models.__dict__[model_name](pretrained=False)
# classes = torch.load(checkpoint)['classes']
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, len(classes))
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

SOS_token = 0
device = torch.device("cuda:0" if True else "cpu")
MAX_LENGTH = 160


class Encoder(nn.Module):
    def __init__(self, feature_vec_size=1024, img_feature_size=50):
        super(Encoder, self).__init__()

        # Load pre-trained ResNet-152 model
        resnet = models.resnet152(pretrained=True)

        # Extract Layers till layer4
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # Upsampling + Convolution for feature maps
        self.upsample = nn.Upsample(size=(img_feature_size, img_feature_size), mode='bilinear',
                                    align_corners=True)  # Upsample to desired spatial resolution
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)  # Convert feature depth
        self.conv2 = nn.Conv2d(1024, 256, kernel_size=1)  # Decrease depth to 256

        # Remaining ResNet layers (after layer4)
        self.residual = nn.Sequential(*list(resnet.children())[-2:-1])  # Exclude the final fc layer

        # For the feature vector
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, feature_vec_size)  # ResNet's penultimate layer outputs 2048 features

        # For the final output
        self.fc2 = nn.Linear(feature_vec_size, 12)  # Final fully connected layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)  # Extract Features up to layer4

        # Create img_features
        img_features = self.upsample(x)  # Upsample feature maps
        img_features = self.conv1(img_features)  # Convolution
        img_features = self.conv2(img_features)  # Reduce depth to 256

        # Pass x through remaining ResNet layers
        x = self.residual(x)

        x = self.avgpool(x)  # Global Average Pooling
        x = torch.flatten(x, 1)  # Flatten
        feature_vector = self.fc1(x)  # Fully connected layer to get feature vector

        out = self.fc2(feature_vector)  # Final Output
        out = self.sigmoid(out)

        return img_features, feature_vector.unsqueeze(0), out


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, 64)
        self.gru = nn.GRU(64, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs.permute(1, 0, 2), decoder_hidden, None
        # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden