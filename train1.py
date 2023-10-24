import os
import glob
import torch
import argparse
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import torch
from custom_dataloader import ID_card_DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import tqdm
from Networks import Encoder, DecoderRNN 
#torch.cuda.empty_cache()
import time
import math


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


def train(args, encoder, encoder_optimizer, decoder, decoder_optimizer, device, dataloader1, epoch, start, criterion_mse, criterion_ctc, batch_size, file_name):

    encoder.train()
    decoder.train()
    batch_loss = 0
    pp = 0
    tot_loss_ctc= 0
    tot_loss_mse = 0
    for batch_idx, (images, target) in enumerate(dataloader1):

        images, target = images.to(device), target
        target_encoder = torch.cat((target['person_coordinate'], target['rotation'].unsqueeze(1), target['scale'].unsqueeze(1), target['transport']), dim=1).to(device).float()
        target_decoder = target['encoded_passage'].to(device)
        image_features, feature_vector, output_mse = encoder(images)
        output_ctc, decoder_hidden, _ = decoder(image_features, feature_vector, target_decoder)
        input_lengths = torch.full((batch_size,), 160)  # All logits sequences have length 160
        target_lengths = torch.full((batch_size,), 160)  # All target sequences have length 160
        for i in range(batch_size):
            zero_numbers = (target_decoder[i, :] == 0).sum()
            target_lengths[i] = target_lengths[i] - zero_numbers
        # CTC Loss
        # with torch.backends.cudnn.flags(enabled=False):
        #     loss_ctc = criterion_ctc(output_ctc, target_decoder, input_lengths, target_lengths)
        loss_ctc = nn.functional.ctc_loss(
                    output_ctc,
                    target_decoder,
                    input_lengths,
                    target_lengths,
                    blank=0,
                    reduction='sum',
                    zero_infinity=True,
                )

        loss_mse = criterion_mse(output_mse, target_encoder)
        print(f'ctc loss is {loss_ctc.item()} and mse loss is {loss_mse.item()}')
        # loss_mse.backward(retain_graph=True)
        # encoder_optimizer.step()
        # tot_loss_mse += loss_mse.item()
        loss_ctc.backward()
        decoder_optimizer.step()
        encoder_optimizer.step()
        tot_loss_ctc += loss_ctc.item()

#         input_values = input_data[int(data[i])].to(device)
#         loss = model(**input_values).loss
#         batch_loss += loss
#
#         batch_loss_mean = batch_loss/len(data)
#         optimizer.zero_grad()
#         batch_loss_mean.backward()
#         optimizer.step()
#         tot_loss += batch_loss.item()
#         batch_loss = 0
#
        if batch_idx % args.log_interval == 1 :

            print('Train Epoch on {}: {} [{}/{} ({:.0f}%)]\tLoss ctc: {:.6f}\t Loss mse: {:.6f}'.format(
                file_name, epoch, (batch_idx+1) * batch_size, len(dataloader1)* batch_size,
                       100 * (batch_idx+1) / len(dataloader1),
                       loss_ctc.item(), loss_mse.item()))

    print('Epoch {} at the end of {} final losses\nLoss mse: {:.6f}\t Loss ctc: {:.6f}'.format(epoch, file_name, tot_loss_mse / (len(dataloader1)*batch_size), tot_loss_ctc/ (len(dataloader1)*batch_size)))
    return
#
#
# def evaluation(args, model, device, test_loader, optimizer, val_loss_min, epoch, input_data):
#
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#
#         for batch_idx, (data, target) in enumerate(test_loader):
#
#             data, target = data, target
#             for i in range(len(data)):
#
#                 input_values = input_data[int(data[i])].to(device)
#                 loss = model(**input_values).loss
#                 val_loss += loss.item()
#
#     val_loss = val_loss/len(test_loader.dataset)
#     print('\nValidation loss: {:.6f}\n'.format(val_loss))
#     # save model if validation loss has decreased
#     if val_loss < val_loss_min:
#
#         if args.save_model:
#
#             #filename = 'model_epock_{0}_val_loss_{1}.pt'.format(epoch, val_loss)
#             #torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
#             filename = 'E:/codes_py/speech2text/saved_models/hubert_epoch_{}'.format(epoch)
#             torch.save(model.state_dict(), filename)
#             val_loss_min = val_loss
#         return val_loss_min
#
#     else:
#         return None


def main():
    # argparse = argparse.parse_args()
    parser = argparse.ArgumentParser(description='PyTorch speech2text')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--valid-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.3, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--weight', default=False,
                        help='path of pretrain weights')
    parser.add_argument('--resume', default=False,
                        help='path of resume weights , "./cnn_83.pt" OR "./FC_83.pt" OR False ')

    args = parser.parse_args()
    use_cuda = args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)

    encoder = Encoder(feature_vec_size=1024, img_feature_size=50).to(device)
    decoder = DecoderRNN(hidden_size=1024, output_size=127).to(device)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=4e-4)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=4e-4)

    scheduler_enc = StepLR(encoder_optimizer, step_size=2, gamma=args.gamma)
    scheduler_dec = StepLR(decoder_optimizer, step_size=2, gamma=args.gamma)

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
                encoder.load_state_dict(checkpoint['state_dict_encoder'])
                decoder.load_state_dict(checkpoint['state_dict_decoder'])
                encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
                decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
            except Exception as e:
                print(e)
    for param in encoder.parameters():
        param.requires_grad_(False)
        #
        # #model.config.ctc_loss_reduction = "mean"
    k = 50

    for i in range(1, k):
        list(encoder.parameters())[-i].requires_grad_(True)

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_trainable_parameters(encoder)} trainable parameters')

    def count_parameters(model):
        return sum(p.numel() for p in encoder.parameters())

    print(f'The model has {count_parameters(encoder)} parameters in sum')
    # Verify which layers are unfrozen
    for name, param in encoder.named_parameters():
        print(name, param.requires_grad)

    height, width = 720, 720
    batch_size = args.batch_size
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    files = glob.glob('/home/kasra/kasra_files/data-shenasname/*.CSV') + glob.glob(
        '/home/kasra/kasra_files/data-shenasname/*.csv')
    file_list = []
    for file in files:
        file_list.append([file, file.split('.')[0].replace('metadata', 'files')])

    criterion_mse = nn.MSELoss(reduction='sum')
    criterion_ctc = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)
    val_loss_min = np.Inf
    for epoch in range(1, args.epochs + 1):
        for index, file in tqdm.tqdm(enumerate(file_list)):
            dataset1 = ID_card_DataLoader(image_folder=file[1], label_file=file[0], transform=trans)
            dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True, drop_last=True)
            start = time.time()
            train(args, encoder, encoder_optimizer, decoder, decoder_optimizer, device, dataloader1, epoch, start, criterion_mse, criterion_ctc, args.batch_size, file[1])
            scheduler_enc.step()
            scheduler_dec.step()
            out_loss = evaluation(args, encoder, encoder_optimizer, decoder, decoder_optimizer, device, testloader, epoch, val_loss_min)
            if out_loss is not None:
                val_loss_min = out_loss


if __name__ == '__main__':
    main()

