import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy


SOS_token = 0
device = torch.device("cuda:0" if True else "cpu")
MAX_LENGTH = 160
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # Load pre-trained ResNet-152
        resnet1 = models.resnet152(pretrained=True)
        resnet2 = copy.deepcopy(resnet1)

        # Backbone
        self.backbone = nn.Sequential(*list(resnet1.children())[:-4])

        # Layer for img_features extraction
        self.img_features_layer = list(resnet1.children())[-4]

        # Regression Head
        self.regression_head = nn.Sequential(
            *list(resnet2.children())[-4: -1],  # ResNet's continuation from after img_features_layer
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
            nn.Linear(2048, 116)  # or 512 based on your preference
        )

        # Convolution layers for img_features (after removing upsample)
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=1, padding=0)

    # Weight initialization
        # self.apply(self.initialize_weights)

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


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
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

        return output.float(), hidden.float(), attn_weights

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
        return output.float(), hidden.float()