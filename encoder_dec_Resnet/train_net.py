import os
import glob
import torch
import argparse
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import torch

import tokenizer
from custom_dataloader import ID_card_DataLoader, All_ID_card_DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import tqdm
from Networks_EN import Encoder, FeatureExtractor, AttnDecoderRNN, EncoderRNN
#torch.cuda.empty_cache()
import time
import math


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def replace_tokens(passage):
    passage = passage.replace('a', 'روی کارت ملی')
    passage = passage.replace('b', 'پشت کارت ملی')
    passage = passage.replace('c', 'شناسنامه جدید')
    passage = passage.replace('d', 'شناسنامه قدیم')
    return passage


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def freeze_params(Encoder, Decoder, trainee):

    for param in Encoder.parameters():
        param.requires_grad_(False)

    for param in Decoder.parameters():
        param.requires_grad_(False)

    if trainee == 'regressor':

        for param in Encoder.regression_head.parameters():
            param.requires_grad_(True)
        k = 150
        for i in range(k):
            list(Encoder.regression_head.parameters())[i].requires_grad_(False)


    elif trainee == 'feature_extractor':

        for submodel in [Encoder.img_features_layer, Encoder.feature_head, Encoder.conv1, Encoder.conv2]:
            for param in submodel.parameters():
                param.requires_grad_(True)

        k = 150
        for i in range(k):
            list(Encoder.img_features_layer.parameters())[i].requires_grad_(False)

        for param in Decoder.parameters():
            param.requires_grad_(True)
        for param in Encoder.regression_head.parameters():
            param.requires_grad_(False)

    # Verify which layers are unfrozen
    # for name, param in encoder.named_parameters():
    #     print(name, param.requires_grad)
def train(args, encoderCNN, feature_extractor, encoderRNN, decoderRNN, encoderCNN_optimizer
          , feature_extractor_optimizer, encoderRNN_optimizer, decoderRNN_optimizer,
          device, dataloader1, epoch, start, criterion_mse, criterion_ctc, batch_size):
    encoderCNN.train()
    feature_extractor.train()
    encoderRNN.train()
    decoderRNN.train()
    batch_loss = 0
    pp = 0
    tot_loss_ctc = 0
    tot_loss_mse = 0
    group_loss_ctc = 0
    group_loss_mse = 0
    criterion_ctc = criterion_ctc.to(device)
    criterion_mse = criterion_mse.to(device)
    for batch_idx, (images, target) in enumerate(dataloader1):

        try:
            print('Start train model for reducing the ctc loss')
            if batch_idx == 0:

                print(f'The encoderCNN has {count_parameters(encoderCNN)} parameters in sum')
                print(f'The feature_extractor has {count_parameters(feature_extractor)} parameters in sum')
                print(f'The encoderRNN has {count_trainable_parameters(encoderRNN)} trainable parameters')
                print(f'The decoderRNN has {count_trainable_parameters(decoderRNN)} trainable parameters')

            images, target = images.to(device), target
            target_encoder = torch.cat((target['person_coordinate'], target['rotation'].unsqueeze(1), target['scale'].unsqueeze(1), target['transport']), dim=1).to(device).float()
            target_decoder = target['encoded_passage'].to(device)
            feature_map, feature_vector = encoderCNN(images)
            # features = feature_extractor(images)
            encoderRNN_init_hidden = torch.cat((feature_vector.unsqueeze(0), target_encoder.unsqueeze(0)), dim=2)
            encoderRNN_output, decoderRNN_init_hidden = encoderRNN(feature_map.view(batch_size, feature_map.size()[1], feature_map.size()[2] * feature_map.size()[3]).permute(0,2,1)
                                                                   , encoderRNN_init_hidden)
            output_ctc, _, _ = decoderRNN(encoderRNN_output, decoderRNN_init_hidden, None)
            input_lengths = torch.full((batch_size,), 160)  # All logits sequences have length 160
            target_lengths = torch.full((batch_size,), 160)  # All target sequences have length 160

            _, topi = output_ctc.topk(1)
            decoded_ids = topi.squeeze().permute(1, 0)
            for i in range(batch_size):
                zero_numbers = (target_decoder[i, :] == 127).sum()
                target_lengths[i] = target_lengths[i] - zero_numbers
                # zero_numbers1 = (decoded_ids[i, :] == 127).sum()
                # input_lengths[i] = input_lengths[i] - zero_numbers1

            encoderCNN_optimizer.zero_grad()
            # feature_extractor.zero_grad()
            encoderRNN_optimizer.zero_grad()
            decoderRNN_optimizer.zero_grad()
            # CTC Loss
            with torch.backends.cudnn.flags(enabled=False):
                loss_ctc = criterion_ctc(output_ctc, target_decoder, input_lengths, target_lengths)
            # loss_ctc = nn.functional.ctc_loss(
            #             output_ctc,
            #             target_decoder,
            #             input_lengths,
            #             target_lengths,
            #             blank=127,
            #             reduction='sum',
            #             zero_infinity=True,
            #         )

            # loss_mse = criterion_mse(features, target_encoder)
            print(f'ctc loss sum is {loss_ctc.item() * batch_size}')
            # loss_mse.backward()
            # encoder_optimizer.step()
            # tot_loss_mse += loss_mse.item() * batch_size
            tot_loss_ctc += loss_ctc.item() * batch_size
            loss_ctc.backward()
            # loss_mse.backward()
            encoderCNN_optimizer.step()
            # feature_extractor_optimizer.step()
            encoderRNN_optimizer.step()
            decoderRNN_optimizer.step()
            group_loss_ctc += loss_ctc.item() * batch_size
            # group_loss_mse += loss_mse.item() * batch_size

            if (batch_idx +1) % args.log_interval == 0 and (batch_idx != 0):
                print('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss ctc: {:.6f}\t'.format(
                epoch, (batch_idx+1) * batch_size, len(dataloader1)* batch_size,
                        100 * (batch_idx+1) / len(dataloader1),
                        group_loss_ctc/(args.log_interval * batch_size)))
                group_loss_ctc = 0
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"An error occurred while reading: {e}\n")

        torch.cuda.empty_cache()

    tot_loss_mse = 0
    group_loss_mse = 0
    for batch_idx, (images, target) in enumerate(dataloader1):

        try:
            print('Start train model for reducing the MSE loss')
            if batch_idx == 0:

                print(f'The encoderCNN has {count_parameters(encoderCNN)} parameters in sum')
                print(f'The feature_extractor has {count_parameters(feature_extractor)} parameters in sum')
                print(f'The encoderRNN has {count_trainable_parameters(encoderRNN)} trainable parameters')
                print(f'The decoderRNN has {count_trainable_parameters(decoderRNN)} trainable parameters')

            images, target = images.to(device), target
            target_encoder = torch.cat((target['person_coordinate'], target['rotation'].unsqueeze(1), target['scale'].unsqueeze(1), target['transport']), dim=1).to(device).float()
            target_decoder = target['encoded_passage'].to(device)
            # feature_map, feature_vector = encoderCNN(images)
            features = feature_extractor(images)
            # encoderRNN_init_hidden = torch.cat((feature_vector.unsqueeze(0), target_encoder.unsqueeze(0)), dim=2)
            # encoderRNN_output, decoderRNN_init_hidden = encoderRNN(feature_map.view(batch_size, feature_map.size()[1], feature_map.size()[2] * feature_map.size()[3]).permute(0,2,1)
            #                                                        , encoderRNN_init_hidden)
            # output_ctc, _, _ = decoderRNN(encoderRNN_output, decoderRNN_init_hidden, None)
            # input_lengths = torch.full((batch_size,), 160)  # All logits sequences have length 160
            # target_lengths = torch.full((batch_size,), 160)  # All target sequences have length 160

            # _, topi = output_ctc.topk(1)
            # decoded_ids = topi.squeeze().permute(1, 0)
            # for i in range(batch_size):
            #     zero_numbers = (target_decoder[i, :] == 127).sum()
            #     target_lengths[i] = target_lengths[i] - zero_numbers
            #     # zero_numbers1 = (decoded_ids[i, :] == 127).sum()
            #     # input_lengths[i] = input_lengths[i] - zero_numbers1
            #
            # encoderCNN_optimizer.zero_grad()
            feature_extractor_optimizer.zero_grad()
            # encoderRNN_optimizer.zero_grad()
            # decoderRNN_optimizer.zero_grad()
            # CTC Loss
            # with torch.backends.cudnn.flags(enabled=False):
            #     loss_ctc = criterion_ctc(output_ctc, target_decoder, input_lengths, target_lengths)
            # loss_ctc = nn.functional.ctc_loss(
            #             output_ctc,
            #             target_decoder,
            #             input_lengths,
            #             target_lengths,
            #             blank=127,
            #             reduction='sum',
            #             zero_infinity=True,
            #         )

            loss_mse = criterion_mse(features, target_encoder)
            print(f'mse loss sum is {loss_mse.item() * batch_size}')
            # loss_mse.backward()
            # encoder_optimizer.step()
            tot_loss_mse += loss_mse.item() * batch_size
            # tot_loss_ctc += loss_ctc.item() * batch_size
            # loss_ctc.backward()
            loss_mse.backward()
            # encoderCNN_optimizer.step()
            feature_extractor_optimizer.step()
            # encoderRNN_optimizer.step()
            # decoderRNN_optimizer.step()
            # group_loss_ctc += loss_ctc.item() * batch_size
            group_loss_mse += loss_mse.item() * batch_size

            if (batch_idx +1) % args.log_interval == 0 and (batch_idx != 0):
                print('Train Epoch {}: [{}/{} ({:.0f}%)]\t Loss mse: {:.6f}'.format(
                epoch, (batch_idx+1) * batch_size, len(dataloader1)* batch_size,
                        100 * (batch_idx+1) / len(dataloader1),
                        group_loss_mse/(args.log_interval * batch_size)))
                group_loss_ctc = 0
                group_loss_mse = 0
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"An error occurred while reading: {e}\n")

        torch.cuda.empty_cache()

    print('Epoch {} CTC TRAINING COMPLETED. final losses:\nLoss mse: {:.6f}\t Loss ctc: {:.6f}'.format(epoch, tot_loss_mse / (len(dataloader1)*batch_size), tot_loss_ctc/ (len(dataloader1)*batch_size)))
    torch.cuda.empty_cache()

    return tot_loss_mse / (len(dataloader1)*batch_size), tot_loss_ctc/ (len(dataloader1)*batch_size)


def evaluation(args, encoderCNN, feature_extractor, encoderRNN, decoderRNN
              , encoderCNN_optimizer, feature_extractor_optimizer, encoderRNN_optimizer
              , decoderRNN_optimizer, device, test_loader, epoch, mse_loss,
               ctc_loss, batch_size, criterion_mse, criterion_ctc, mse_loss_min, ctc_loss_min):

    encoderCNN.eval()
    feature_extractor.eval()
    encoderRNN.eval()
    decoderRNN.eval()

    val_loss = 0
    with torch.no_grad():

        for batch_idx, (images, target) in enumerate(test_loader):
            try:
                images, target = images.to(device), target
                target_encoder = torch.cat((target['person_coordinate'], target['rotation'].unsqueeze(1),
                                            target['scale'].unsqueeze(1), target['transport']), dim=1).to(device).float()
                target_decoder = target['encoded_passage'].to(device)
                feature_map, feature_vector = encoderCNN(images)
                features = feature_extractor(images)
                encoderRNN_init_hidden = torch.cat((feature_vector.unsqueeze(0), target_encoder.unsqueeze(0)), dim=2)
                encoderRNN_output, decoderRNN_init_hidden = encoderRNN(feature_map.view(batch_size, feature_map.size()[1], feature_map.size()[2] * feature_map.size()[3]).permute(0,2,1)
                                                                       , encoderRNN_init_hidden)
                output_ctc, _, _ = decoderRNN(encoderRNN_output, decoderRNN_init_hidden, None)
                input_lengths = torch.full((batch_size,), 160)  # All logits sequences have length 160
                target_lengths = torch.full((batch_size,), 160)  # All target sequences have length 160

                print('target features are:\n{}'.format(target_encoder))
                print('output featurells are:\n{}'.format(features))
                _, topi = output_ctc.topk(1)
                decoded_ids = topi.squeeze(2).permute(1, 0)

                for i in range(batch_size):
                    zero_numbers = (target_decoder[i, :] == 127).sum()
                    target_lengths[i] = target_lengths[i] - zero_numbers

                # CTC Loss
                with torch.backends.cudnn.flags(enabled=False):
                    loss_ctc = criterion_ctc(output_ctc, target_decoder, input_lengths, target_lengths)

                loss_mse = criterion_mse(features, target_encoder)
                print(f'ctc mean loss is {loss_ctc.item()} and mse mean loss is {loss_mse.item()}')
                print('target passage is: \n{}'.format(replace_tokens(target['passage'][0])))


                decoded_words = []
                for vector in decoded_ids:
                    for idx in vector:
                        if idx.item() == 1:
                            break
                        decoded_words.append(tokenizer.ind2char(idx.item()))

                output_passage = ''.join(char for char in decoded_words)
                print('output passage is: \n{}'.format(replace_tokens(output_passage)))
            except Exception as e:
                print(f"An error occurred: {e}\n")



        torch.cuda.empty_cache()
        if args.save_model and (mse_loss < mse_loss_min):

            filename = ('/home/kasra/PycharmProjects/Larkimas/encoder_decoder/model_checkpoints/feature_extractor: epoch: {0} mse_l:{1}.pt').format(epoch, np.round(mse_loss, 3))
            torch.save({'epoch': epoch, 'state_dict feature_extractor': feature_extractor.state_dict(),
                        'feature_extractor_optimizer': feature_extractor_optimizer.state_dict()}, filename)
            print('feature extractor model has been saved in {}'.format(filename))
            mse_loss_min = mse_loss

        if args.save_model and (ctc_loss < ctc_loss_min):
            filename = ('/home/kasra/PycharmProjects/Larkimas/encoder_decoder/model_checkpoints'
                        '/decoder: epoch: {0} ctc_l:{1}.pt').format(epoch, np.round(ctc_loss, 3))
            torch.save({'epoch': epoch, 'state_dict encoderCNN': encoderCNN.state_dict(),
                        'state_dict encoderRNN': encoderRNN.state_dict(), 'state_dict decoderRNN': decoderRNN.state_dict(),
                        'enoderCNN_optimizer': encoderCNN_optimizer.state_dict(), 'enoderRNN_optimizer': encoderRNN_optimizer.state_dict()
                        , 'decoderRNN_optimizer': decoderRNN_optimizer.state_dict()}, filename)
            print('decoder model has been saved in {}'.format(filename))
            ctc_loss_min = ctc_loss

    return mse_loss_min, ctc_loss_min

#
#     else:
#         return None


def main():
    # argparse = argparse.parse_args()
    parser = argparse.ArgumentParser(description='PyTorch speech2text')
    parser.add_argument('--batch-size', type=int, default=6, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--valid-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.4, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--weight', default=False,
                        help='path of pretrain weights')
    parser.add_argument('--resume', default=False,
                        help='path of resume weights , "./cnn_83.pt" OR "./FC_83.pt" OR False ')
# '/home/kasra/PycharmProjects/Larkimas/model_checkpoints/encoder_epoch_1_ctc_loss_4.3.pt'
    args = parser.parse_args()
    use_cuda = args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)

    encoderCNN = Encoder().to(device)
    encoderRNN = EncoderRNN(input_size=128, hidden_size=128, dropout_p=0).to(device)
    decoderRNN = AttnDecoderRNN(hidden_size=128, output_size=128, dropout_p=0).to(device)

    encoderCNN_optimizer = torch.optim.Adam(encoderCNN.parameters(), lr=args.lr)
    decoderRNN_optimizer = torch.optim.Adam(decoderRNN.parameters(), lr=args.lr)
    encoderRNN_optimizer = torch.optim.Adam(encoderRNN.parameters(), lr=args.lr)
    # , weight_decay=4e-4
    scheduler_enc = StepLR(encoderCNN_optimizer, step_size=1, gamma=args.gamma)
    scheduler_dec = StepLR(feature_extractor_optimizer, step_size=1, gamma=args.gamma)
    scheduler_enc = StepLR(encoderRNN_optimizer, step_size=1, gamma=args.gamma)
    scheduler_enc = StepLR(decoderRNN_optimizer, step_size=1, gamma=args.gamma)

    if args.weight:
        if os.path.isfile(args.weight):
            checkpoint = torch.load(args.weight)
            try:
                encoderCNN.load_state_dict(checkpoint['state_dict_encoderCNN'])
                encoderRNN.load_state_dict(checkpoint['state_dict_encoderRNN'])
                decoderRNN.load_state_dict(checkpoint['state_dict_decoderRNN'])
            except Exception as e:
                print(e)

    # args.resume = False
    if args.resume:
        if os.path.isfile(args.resume):
            # checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            checkpoint = torch.load(args.resume)
            try:
                args.start_epoch = checkpoint['epoch']
                encoderCNN.load_state_dict(checkpoint['state_dict_encoderCNN'])
                feature_extractor.load_state_dict(checkpoint['state_dict_feature_extractor'])
                encoderRNN.load_state_dict(checkpoint['state_dict_encoderRNN'])
                decoderRNN.load_state_dict(checkpoint['state_dict_decoderRNN'])
                encoderCNN_optimizer.load_state_dict(checkpoint['encoderCNN_optimizer'])
                feature_extractor_optimizer.load_state_dict(checkpoint['feature_extractor_optimizer'])
                encoderRNN_optimizer.load_state_dict(checkpoint['encoderRNN_optimizer'])
                decoderRNN_optimizer.load_state_dict(checkpoint['decoderRNN_optimizer'])
            except Exception as e:
                print(e)
    # for param in encoder.parameters():
    #     param.requires_grad_(False)
    #     #
    #     # #model.config.ctc_loss_reduction = "mean"
    # k = 350
    #
    # for i in range(1, k):
    #     list(encoder.parameters())[-i].requires_grad_(True)
    #
    # def count_trainable_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
    #
    # print(f'The model has {count_trainable_parameters(encoder)} trainable parameters')
    #
    # def count_parameters(model):
    #     return sum(p.numel() for p in encoder.parameters())
    #
    # print(f'The model has {count_parameters(encoder)} parameters in sum')
    # # Verify which layers are unfrozen
    # for name, param in encoder.named_parameters():
    #     print(name, param.requires_grad)

    height, width = 480, 480
    batch_size = args.batch_size
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    # /home/kasra/kasra_files/data-shenasname/
    files = glob.glob('E:/codes_py/Larkimas/Data_source/UBUNTU 20_0/*.csv')
    file_list = []
    for file in files:
        file_list.append([file, file.split('.')[0].replace('metadata', 'files')])

    files_test = glob.glob('/home/kasra/kasra_files/data-shenasname/validation_file/*.csv')
    file_test = []
    for file in files_test:
        file_test.append([file, file.split('.')[0]])

    criterion_mse = nn.MSELoss(reduction='mean')
    criterion_ctc = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    mse_loss_min = np.Inf
    ctc_loss_min = np.Inf
    dataset1 = All_ID_card_DataLoader(data_file='/home/kasra/kasra_files/data-shenasname/data_loc', transform=trans)
    dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True, drop_last=True)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        mse_loss, ctc_loss = train(args, encoderCNN, feature_extractor, encoderRNN, decoderRNN
                                   , encoderCNN_optimizer, feature_extractor_optimizer
                                   , encoderRNN_optimizer, decoderRNN_optimizer,device, dataloader1
                                   , epoch, start, criterion_mse, criterion_ctc, args.batch_size)

        for index, file in tqdm.tqdm(enumerate(file_test)):
            dataset = ID_card_DataLoader(image_folder=file[1], label_file=file[0], transform=trans)
            test_loader = DataLoader(dataset, batch_size=args.valid_batch_size, shuffle=True, drop_last=True)
            mse_loss_min, ctc_loss_min = evaluation(args, encoderCNN, feature_extractor, encoderRNN, decoderRNN
                                   , encoderCNN_optimizer, feature_extractor_optimizer, encoderRNN_optimizer
                                   , decoderRNN_optimizer, device, test_loader, epoch, mse_loss, ctc_loss
                                   , args.valid_batch_size, criterion_mse, criterion_ctc, mse_loss_min, ctc_loss_min)

        scheduler_enc.step()
        scheduler_dec.step()

if __name__ == '__main__':
    main()

