import os

from data_loader import UnetDataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
import glob
import pickle
import tqdm
import cv2
import numpy as np

height, width = 720, 720
batch_size = 16
trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

files = os.listdir('/home/kasra/kasra_files/data-shenasname')
file_list = []
for file in files:
    if file not in  glob.glob('/home/kasra/kasra_files/data-shenasname/*.csv'):
        if file.split('_')[-1] == 'A':
            file_list.append([file, '_'.join(file.split('_')[:-1])])
tokens = []
max = 0
min = 100

for index, file in tqdm.tqdm(enumerate(file_list)):

    dataset1 = UnetDataLoader(transformed_image_file='/home/kasra/kasra_files/data-shenasname/' + file[0], original_image_file='/home/kasra/kasra_files/data-shenasname/' + file[1], transform=trans)
    dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True, drop_last=True)

    # Iterate through the DataLoader to get one dictionary for each data point
    for batch in dataloader1:
        images, labels = batch  # Unpack the batch into images and labels
