
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
import copy

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

        return img_features.float(), feature_vector.float().unsqueeze(0), out.float()


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
                if i < target_tensor.shape[1]:
                    decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
                else:
                    decoder_input = (127 * torch.ones_like(target_tensor[:, 0])).unsqueeze(1)

            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs.float().permute(1, 0, 2), decoder_hidden.float(), None
        # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        keys = keys.view(keys.shape[0], keys.shape[1], keys.shape[2]*keys.shape[3] ).permute(0, 2 ,1)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size = 512, output_size = 128, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs.permute(1, 0, 2), decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        # Load pre-trained ResNet-152
        resnet1 = models.resnet152(pretrained=True)
        resnet2 = copy.deepcopy(resnet1)

        # Backbone
        self.backbone = nn.Sequential(*list(resnet1.children())[:-4])

        # Layer for img_features extraction
        self.img_features_layer = list(resnet1.children())[-4]

        # Regression Head
        self.regression_head = nn.Sequential(
            *list(resnet2.children())[-4: -1],  # ResNet continuation from after img_features_layer
            nn.Flatten(),
            nn.Linear(2048, 256),  # Additional layer for extracting more complex features
            nn.ReLU(),
            nn.Linear(256, 12),
            nn.ReLU()
        )

        # Feature vector head
        self.feature_head = nn.Sequential(
            *list(resnet1.children())[-3: -1],  # ResNet's continuation from after img_features_layer
            nn.Flatten(),
            nn.Linear(2048, 500)  # or 512 based on your preference
        )

        # Convolution layers for img_features (after removing upsample)
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=1, padding=0)

    def forward(self, x):
        # Backbone
        x = self.backbone(x)

        # Extract img_features
        img_features_input = self.img_features_layer(x)
        img_features = self.conv1(img_features_input)
        img_features = self.conv2(img_features)

        # Regression Head
        regression_out = self.regression_head(x)

        # Feature vector head
        feature_out = self.feature_head(img_features_input)

        return img_features.float(), feature_out.float().unsqueeze(0), regression_out.float()
