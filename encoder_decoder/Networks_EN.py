import torch
import torch.nn as nn
import torch.nn.functional as F


SOS_token = 0
device = torch.device("cuda:0" if True else "cpu")
MAX_LENGTH = 160
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
            nn.Conv2d(128, 64, kernel_size=7, stride=3, padding=3),  # This will reduce 90x90 to 30x30
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        # Calculate the size of the feature map after downsampling
        downsampled_output_size = (30, 30)  # This is the expected output size after downsampling

        # Intermediate dense layer to extract more complex features
        self.intermediate_dense = nn.Linear(downsampled_output_size[0] * downsampled_output_size[1] * 64, 512)

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
            nn.init.constant_(m.bias, 0)


class FeatureExtractor(nn.Module):
    def __init__(self, base_filters=64, reduced_filters=64, output_features=12):
        super(FeatureExtractor, self).__init__()

        # Define pathways with kernel sizes 3, 7, and 13, each with 3 layers
        self.pathway1 = self._make_layers(base_filters, kernel_size=3, num_layers=3)
        self.pathway2 = self._make_layers(base_filters, kernel_size=7, num_layers=3)
        self.pathway3 = self._make_layers(base_filters, kernel_size=13, num_layers=3)

        # Reduction layer to fewer channels and higher stride to reduce size
        self.reduce_size_and_channels = nn.Sequential(
            nn.Conv2d(base_filters * 3, reduced_filters, kernel_size=3, stride=2, padding=1),
            # This will reduce size roughly to half
            nn.BatchNorm2d(reduced_filters),
            nn.LeakyReLU()
        )

        # Assuming the pathways above reduce the size to 1/4 of the original,
        # and the reduce_size_and_channels layer halves it again,
        # we get a feature map of size 45x45 if the original size is 360x360.

        # Final layer to output a vector with the size of 1x12
        self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(reduced_filters * 45 * 45, output_features),
            nn.ReLU()  # Ensures all output features are non-negative
        )

        # Weight initialization
        self.apply(self.initialize_weights)

    def _make_layers(self, in_channels, kernel_size, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm2d(in_channels))
            layers.append(nn.LeakyReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        # Apply each pathway
        p1 = self.pathway1(x)
        p2 = self.pathway2(x)
        p3 = self.pathway3(x)

        # Concatenate the outputs along the depth
        concatenated_features = torch.cat((p1, p2, p3), 1)

        # Reduce size and channels
        reduced_features = self.reduce_size_and_channels(concatenated_features)

        # Flatten and pass through the final layer to get the feature vector
        feature_vector = self.final_layer(reduced_features)

        # Note: feature_vector elements will be non-negative due to ReLU,
        # but not necessarily <= 1. If needed, add scaling here.

        return feature_vector

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
            nn.init.constant_(m.bias, 0)



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
    def __init__(self, hidden_size = 128, output_size = 128, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.output_size = output_size
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
        # embedded = self.one_hot_encode(input)
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

    def one_hot_encode(self, input_tensor):
        # Assuming input_tensor contains integer values from 0 to 127
        batch_size, seq_len= input_tensor.size()
        one_hot = torch.zeros(batch_size, seq_len, self.output_size, device=device)
        one_hot = one_hot.scatter(2, input_tensor.unsqueeze(2).long(), 1)  # Fill with 1s at specified indices
        return one_hot

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, initial_hidden=None):
        output, hidden = self.gru(input, initial_hidden)
        return output, hidden