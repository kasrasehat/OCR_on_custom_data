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
from custom_dataloader import ID_card_DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import tqdm
from Networks import Encoder, DecoderRNN, CustomModel, AttnDecoderRNN
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
def train(args, encoder, encoder_optimizer, decoder, decoder_optimizer, device, dataloader1, epoch, start, criterion_mse, criterion_ctc, batch_size, file_name):

    encoder.train()
    decoder.train()
    batch_loss = 0
    pp = 0
    tot_loss_ctc = 0
    tot_loss_mse = 0
    group_loss_ctc = 0
    group_loss_mse = 0
    criterion_ctc = criterion_ctc.to(device)
    for batch_idx, (images, target) in enumerate(dataloader1):

        try:
            print('Start train model for reducing the ctc loss')
            if batch_idx == 0:
                freeze_params(encoder, decoder, 'feature_extractor')
                for name, param in encoder.named_parameters():
                    print(name, param.requires_grad)
                print(f'The encoder has {count_parameters(encoder)} parameters in sum')
                print(f'The decoder has {count_parameters(decoder)} parameters in sum')
                print(f'The encoder has {count_trainable_parameters(encoder)} trainable parameters')
                print(f'The decoder has {count_trainable_parameters(decoder)} trainable parameters')

            images, target = images.to(device), target
            target_encoder = torch.cat((target['person_coordinate'], target['rotation'].unsqueeze(1), target['scale'].unsqueeze(1), target['transport']), dim=1).to(device).float()
            target_decoder = target['encoded_passage'].to(device)
            image_features, feature_vector, output_mse = encoder(images)
            decoder_input = torch.cat((feature_vector, target_encoder.unsqueeze(0)), dim=2)
            output_ctc, decoder_hidden, _ = decoder(image_features, decoder_input, None)
            input_lengths = torch.full((batch_size,), 160)  # All logits sequences have length 160
            target_lengths = torch.full((batch_size,), 160)  # All target sequences have length 160

            _, topi = output_ctc.topk(1)
            decoded_ids = topi.squeeze().permute(1, 0)
            for i in range(batch_size):
                zero_numbers = (target_decoder[i, :] == 127).sum()
                target_lengths[i] = target_lengths[i] - zero_numbers
                # zero_numbers1 = (decoded_ids[i, :] == 127).sum()
                # input_lengths[i] = input_lengths[i] - zero_numbers1

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
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

            loss_mse = criterion_mse(output_mse, target_encoder)
            print(f'ctc loss sum is {loss_ctc.item() * batch_size} and mse loss sum is {loss_mse.item() * batch_size}')
            # loss_mse.backward()
            # encoder_optimizer.step()
            tot_loss_mse += loss_mse.item() * batch_size
            loss_ctc.backward()
            decoder_optimizer.step()
            encoder_optimizer.step()
            tot_loss_ctc += loss_ctc.item() * batch_size
            group_loss_ctc += loss_ctc.item() * batch_size
            group_loss_mse += loss_mse.item() * batch_size

            if (batch_idx +1) % args.log_interval == 0 and (batch_idx != 0):
                print('Train Epoch on {}: {} [{}/{} ({:.0f}%)]\tLoss ctc: {:.6f}\t Loss mse: {:.6f}'.format(
                file_name, epoch, (batch_idx+1) * batch_size, len(dataloader1)* batch_size,
                        100 * (batch_idx+1) / len(dataloader1),
                        group_loss_ctc/(args.log_interval * batch_size), group_loss_mse/(args.log_interval * batch_size)))
                group_loss_ctc = 0
                group_loss_mse = 0
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"An error occurred while reading {file_name}: {e}\n")

        torch.cuda.empty_cache()
    print('Epoch {} CTC TRAINING at the end of {} COMPLETED. final losses:\nLoss mse: {:.6f}\t Loss ctc: {:.6f}'.format(epoch, file_name, tot_loss_mse / (len(dataloader1)*batch_size), tot_loss_ctc/ (len(dataloader1)*batch_size)))
    torch.cuda.empty_cache()

    batch_loss = 0
    pp = 0
    tot_loss_ctc = 0
    tot_loss_mse = 0
    group_loss_ctc = 0
    group_loss_mse = 0
    for batch_idx, (images, target) in enumerate(dataloader1):

        try:
            freeze_params(encoder, decoder, 'regressor')
            print('Start train model for reducing the mse loss')
            if batch_idx == 0:
                for name, param in encoder.named_parameters():
                    print(name, param.requires_grad)
                print(f'The encoder has {count_parameters(encoder)} parameters in sum')
                print(f'The decoder has {count_parameters(decoder)} parameters in sum')
                print(f'The encoder has {count_trainable_parameters(encoder)} trainable parameters')
                print(f'The decoder has {count_trainable_parameters(decoder)} trainable parameters')

            images, target = images.to(device), target
            target_encoder = torch.cat((target['person_coordinate'], target['rotation'].unsqueeze(1),
                                        target['scale'].unsqueeze(1), target['transport']), dim=1).to(device).float()
            target_decoder = target['encoded_passage'].to(device)
            image_features, feature_vector, output_mse = encoder(images)
            decoder_input = torch.cat((feature_vector, target_encoder.unsqueeze(0)), dim=2)
            output_ctc, decoder_hidden, _ = decoder(image_features, decoder_input, None)
            input_lengths = torch.full((batch_size,), 160)  # All logits sequences have length 160
            target_lengths = torch.full((batch_size,), 160)  # All target sequences have length 160
            _, topi = output_ctc.topk(1)
            decoded_ids = topi.squeeze().permute(1, 0)
            for i in range(batch_size):
                zero_numbers = (target_decoder[i, :] == 127).sum()
                target_lengths[i] = target_lengths[i] - zero_numbers

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
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

            loss_mse = criterion_mse(output_mse, target_encoder)
            print(f'ctc loss sum is {loss_ctc.item() * batch_size} and mse loss sum is {loss_mse.item() * batch_size}')
            loss_mse.backward()
            encoder_optimizer.step()
            tot_loss_mse += loss_mse.item() * batch_size
            # loss_ctc.backward()
            # decoder_optimizer.step()
            # encoder_optimizer.step()
            tot_loss_ctc += loss_ctc.item() * batch_size
            group_loss_ctc += loss_ctc.item() * batch_size
            group_loss_mse += loss_mse.item() * batch_size

            if (batch_idx + 1) % args.log_interval == 0 and (batch_idx != 0):
                print('Train Epoch on {}: {} [{}/{} ({:.0f}%)]\tLoss ctc: {:.6f}\t Loss mse: {:.6f}'.format(
                    file_name, epoch, (batch_idx + 1) * batch_size, len(dataloader1) * batch_size,
                                      100 * (batch_idx + 1) / len(dataloader1),
                                      group_loss_ctc / (args.log_interval * batch_size),
                                      group_loss_mse / (args.log_interval * batch_size)))
                group_loss_ctc = 0
                group_loss_mse = 0
        except Exception as e:
            print(f"An error occurred while reading {file_name}: {e}\n")

        # torch.cuda.empty_cache()
    print('Epoch {} MSE TRAINING at the end of {} COMPLETED. final losses:\nLoss mse: {:.6f}\t Loss ctc: {:.6f}'.format(
        epoch, file_name, tot_loss_mse / (len(dataloader1) * batch_size),
        tot_loss_ctc / (len(dataloader1) * batch_size)))
    torch.cuda.empty_cache()
    return tot_loss_mse / (len(dataloader1)*batch_size), tot_loss_ctc/ (len(dataloader1)*batch_size)


def evaluation(args, encoder, decoder, device, test_loader,
               epoch, mse_loss, ctc_loss, batch_size, criterion_mse, criterion_ctc,
               mse_loss_min, ctc_loss_min, encoder_optimizer, decoder_optimizer):

    encoder.eval()
    decoder.eval()
    val_loss = 0
    with torch.no_grad():

        for batch_idx, (images, target) in enumerate(test_loader):
            try:
                images, target = images.to(device), target
                target_encoder = torch.cat((target['person_coordinate'], target['rotation'].unsqueeze(1),
                                            target['scale'].unsqueeze(1), target['transport']), dim=1).to(device).float()

                target_decoder = target['encoded_passage'].to(device)
                image_features, feature_vector, output_mse = encoder(images)

                print('target features are:\n{}'.format(target_encoder))
                print('output featurells are:\n{}'.format(output_mse))
                decoder_input = torch.cat((feature_vector, output_mse.unsqueeze(0)), dim = 2)
                output_ctc, decoder_hidden, _ = decoder(image_features, decoder_input, None)
                _, topi = output_ctc.topk(1)
                decoded_ids = topi.squeeze(2).permute(1, 0)
                input_lengths = torch.full((batch_size,), 160)  # All logits sequences have length 160
                target_lengths = torch.full((batch_size,), 160)  # All target sequences have length 160

                for i in range(batch_size):
                    zero_numbers = (target_decoder[i, :] == 127).sum()
                    target_lengths[i] = target_lengths[i] - zero_numbers

                # CTC Loss
                with torch.backends.cudnn.flags(enabled=False):
                    loss_ctc = criterion_ctc(output_ctc, target_decoder, input_lengths, target_lengths)

                loss_mse = criterion_mse(output_mse, target_encoder)
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


#     val_loss = val_loss/len(test_loader.dataset)
#     print('\nValidation loss: {:.6f}\n'.format(val_loss))
#     # save model if validation loss has decreased
#     if val_loss < val_loss_min:
        torch.cuda.empty_cache()
        if args.save_model and (mse_loss < mse_loss_min or ctc_loss < ctc_loss_min):

            filename = ('/home/kasra/PycharmProjects/Larkimas/model_checkpoints'
                        '/epoch_{0}_ctc_l_{:.3f}_mse_l{:.3f}.pt').format(epoch, ctc_loss, mse_loss)
            torch.save({'epoch': epoch, 'state_dict encoder': encoder.state_dict(),
                        'encoder_optimizer': encoder_optimizer.state_dict(), 'state_dict decoder': decoder.state_dict(),
                        'decoder_optimizer': decoder_optimizer.state_dict()}, filename)
            print('model has been saved in {}'.format(filename))
            mse_loss_min = mse_loss
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
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
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

    encoder = CustomModel().to(device)
    decoder = AttnDecoderRNN(hidden_size=128, output_size=128).to(device)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    # , weight_decay=4e-4
    scheduler_enc = StepLR(encoder_optimizer, step_size=1, gamma=args.gamma)
    scheduler_dec = StepLR(decoder_optimizer, step_size=1, gamma=args.gamma)

    if args.weight:
        if os.path.isfile(args.weight):
            checkpoint = torch.load(args.weight)
            try:
                encoder.load_state_dict(checkpoint['state_dict_encoder'])
                decoder.load_state_dict(checkpoint['state_dict_decoder'])
            except Exception as e:
                print(e)

    # args.resume = False
    if args.resume:
        if os.path.isfile(args.resume):
            # checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            checkpoint = torch.load(args.resume)
            try:
                args.start_epoch = checkpoint['epoch']
                encoder.load_state_dict(checkpoint['state_dict encoder'])
                decoder.load_state_dict(checkpoint['state_dict decoder'])
                encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
                decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
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

    height, width = 720, 720
    batch_size = args.batch_size
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    # /home/kasra/kasra_files/data-shenasname/
    files = glob.glob('/home/kasra/kasra_files/data-shenasname/*.CSV') + glob.glob(
        '/home/kasra/kasra_files/data-shenasname/*.csv')
    file_list = []
    for file in files:
        file_list.append([file, file.split('.')[0].replace('metadata', 'files')])

    files_test = glob.glob('/home/kasra/kasra_files/data-shenasname/validation_file/*.CSV') + glob.glob(
        '/home/kasra/kasra_files/data-shenasname/validation_file/*.csv')
    file_test = []
    for file in files_test:
        file_test.append([file, file.split('.')[0]])

    criterion_mse = nn.MSELoss(reduction='mean')
    criterion_ctc = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    mse_loss_min = np.Inf
    ctc_loss_min = np.Inf
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        for index, file in tqdm.tqdm(enumerate(file_list)):
            dataset1 = ID_card_DataLoader(image_folder=file[1], label_file=file[0], transform=trans)
            dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True, drop_last=True)
            mse_loss, ctc_loss = train(args, encoder, encoder_optimizer, decoder, decoder_optimizer, device, dataloader1
                                      , epoch, start, criterion_mse, criterion_ctc, args.batch_size, file[1])
            for index, file in tqdm.tqdm(enumerate(file_test)):
                dataset = ID_card_DataLoader(image_folder=file[1], label_file=file[0], transform=trans)
                test_loader = DataLoader(dataset, batch_size=args.valid_batch_size, shuffle=True, drop_last=True)
                mse_loss_min, ctc_loss_min = evaluation(args, encoder, decoder, device, test_loader,
                                      epoch, mse_loss, ctc_loss, args.valid_batch_size, criterion_mse, criterion_ctc,
                                      mse_loss_min, ctc_loss_min, encoder_optimizer, decoder_optimizer)

        scheduler_enc.step()
        scheduler_dec.step()

if __name__ == '__main__':
    main()

