import os
from data_loader import UnetDataLoader
import glob
import torch
import argparse
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import tqdm
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
def train(args, model, optimizer, device, dataloader1, epoch, start, criterion_mse, batch_size, file_name):

    model.train()
    batch_loss = 0
    pp = 0
    tot_loss_ctc = 0
    tot_loss_mse = 0
    group_loss_ctc = 0
    group_loss_mse = 0
    criterion_mse = criterion_mse.to(device)
    for batch_idx, (transformed_image, original_image) in enumerate(dataloader1):

        try:
            input, target = transformed_image.to(device), original_image.to(device)
            output = model(input)
            loss_mse = criterion_mse(output, target)
            print(f'mse loss sum is {loss_mse.item() * batch_size}')
            optimizer.zero_grad()
            loss_mse.backward()
            optimizer.step()
            tot_loss_mse += loss_mse.item() * batch_size
            group_loss_mse += loss_mse.item() * batch_size

            if (batch_idx +1) % args.log_interval == 0 and (batch_idx != 0):
                print('Train Epoch on {}: {} [{}/{} ({:.0f}%)]\t Loss mse: {:.6f}'.format(
                file_name, epoch, (batch_idx+1) * batch_size, len(dataloader1)* batch_size,
                        100 * (batch_idx+1) / len(dataloader1), group_loss_mse/(args.log_interval * batch_size)))
                group_loss_ctc = 0
                group_loss_mse = 0
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"An error occurred while reading {file_name}: {e}\n")

    torch.cuda.empty_cache()
    print('Epoch {} MSE TRAINING at the end of {} COMPLETED. final losses:\nLoss mse: {:.6f}'.format(epoch, file_name, tot_loss_mse / (len(dataloader1)*batch_size)))
    torch.cuda.empty_cache()

    return tot_loss_mse / (len(dataloader1)*batch_size)


def evaluation(args, model, device, test_loader,
               epoch, mse_loss, batch_size, criterion_mse,
               mse_loss_min, optimizer):

    model.eval()
    val_loss = 0
    with torch.no_grad():

        for batch_idx, (transformed_image, original_image) in enumerate(test_loader):
            try:
                input, target = transformed_image.to(device), original_image.to(device)
                output = model(input)
                loss_mse = criterion_mse(output, target)
                print(f'mse mean loss is {loss_mse.item()}')

            except Exception as e:
                print(f"An error occurred: {e}\n")


        val_loss = loss_mse.item()
        print('\nValidation loss: {:.6f}\n'.format(val_loss))

        torch.cuda.empty_cache()
        if args.save_model and (val_loss < mse_loss_min):

            filename = ('E:/codes_py/Larkimas/Unet/model_checkpoints'
                        '/epoch_{0}_mse_l: {1}.pt').format(epoch, np.round(mse_loss, 3))
            torch.save({'epoch': epoch, 'state_dict_model': model.state_dict(),
                        'optimizer': optimizer.state_dict()}, filename)
            print('model has been saved in {}'.format(filename))
            mse_loss_min = val_loss
        return mse_loss_min

#
#     else:
#         return None


def main():
    # argparse = argparse.parse_args()
    parser = argparse.ArgumentParser(description='PyTorch speech2text')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--valid-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.2, metavar='M',
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

    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=3, out_channels=3, init_features=48, pretrained=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # , weight_decay=4e-4
    scheduler_enc = StepLR(optimizer, step_size=1, gamma=args.gamma)

    if args.weight:
        if os.path.isfile(args.weight):
            checkpoint = torch.load(args.weight)
            try:
                model.load_state_dict(checkpoint['state_dict_model'])
            except Exception as e:
                print(e)

    # args.resume = False
    if args.resume:
        if os.path.isfile(args.resume):
            # checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            checkpoint = torch.load(args.resume)
            try:
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict_model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
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

    # /home/kasra/kasra_files/data-shenasname
    files = os.listdir('E:/codes_py/Larkimas/Data_source/UBUNTU 20_0')
    file_list = []
    for file in files:
        if file not in glob.glob('E:/codes_py/Larkimas/Data_source/UBUNTU 20_0/*.csv'):
            if file.split('_')[-1] == 'A':
                file_list.append([file, '_'.join(file.split('_')[:-1])])
    criterion_mse = nn.MSELoss(reduction='mean')
    mse_loss_min = np.Inf
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        for index, file in tqdm.tqdm(enumerate(file_list)):

            dataset1 = UnetDataLoader(transformed_image_file='E:/codes_py/Larkimas/Data_source/UBUNTU 20_0/' + file[0],
                                      original_image_file='E:/codes_py/Larkimas/Data_source/UBUNTU 20_0/' + file[1],
                                      transform=trans)
            dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True, drop_last=True)
            mse_loss = train(args, model, optimizer, device, dataloader1
                                      , epoch, start, criterion_mse, args.batch_size, file[1])


            dataset = UnetDataLoader(transformed_image_file='E:/codes_py/Larkimas/Data_source/UBUNTU 20_0/unet_test/transformed_images',
                                      original_image_file='E:/codes_py/Larkimas/Data_source/UBUNTU 20_0/unet_test/original_images',
                                      transform=trans)
            test_loader = DataLoader(dataset, batch_size=args.valid_batch_size, shuffle=True, drop_last=True)
            mse_loss_min = evaluation(args, model, device, test_loader,
                                      epoch, mse_loss, args.valid_batch_size, criterion_mse,
                                      mse_loss_min, optimizer)
        scheduler_enc.step()


if __name__ == '__main__':
    main()

