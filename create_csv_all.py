from custom_dataloader import ID_card_DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
import glob
import pickle
import tqdm
import cv2
import numpy as np
import os
import pandas as pd


def replace_tokens(passage):
    passage = passage.replace('a', 'روی کارت ملی')
    passage = passage.replace('b', 'پشت کارت ملی')
    passage = passage.replace('c', 'شناسنامه جدید')
    passage = passage.replace('d', 'شناسنامه قدیم')
    return passage

def create_csv_if_not_exists(csv_path):
    if not os.path.exists(csv_path):
        # Create an empty DataFrame with the same columns as BB
        df = pd.DataFrame(columns=['location','csv_file'])

        # Save the empty DataFrame to a new CSV file named "AAA.csv"
        df.to_csv(csv_path, encoding='UTF-8-SIG', index=False)
        print(f"csv file '{csv_path}' created.")
    else:
        print(f"csv file '{csv_path}' already exists.")


def display_image(batch, index):
    # Extract the ith image
    img = batch[index].numpy()

    # Convert from CHW to HWC
    img = np.transpose(img, (1, 2, 0))
    img = (img + np.ones_like(img))/2

    # Convert pixel values from [0, 1] to [0, 255]
    img = (img * 255).astype(np.uint8)

    # Display the image using cv2
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


height, width = 720, 720
batch_size = 16
trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


csv_path = '/home/kasra/kasra_files/data-shenasname/data_loc'
create_csv_if_not_exists(csv_path)
new_csv = pd.read_csv(csv_path, encoding='UTF-8-SIG')
files = glob.glob('/home/kasra/kasra_files/data-shenasname/*.CSV') + glob.glob('/home/kasra/kasra_files/data-shenasname/*.csv')
file_list = []
for file in files:
    file_list.append([file, file.split('.')[0].replace('metadata', 'files')])
tokens = []
max = 0
min = 100
p = 0
for index, file in tqdm.tqdm(enumerate(file_list)):

    for i, image_file in enumerate(os.listdir(file[1])):
        new_csv.loc[p, 'location'] = file[1] + f'/{image_file}'
        new_csv.loc[p, 'csv_file'] = file[0]
        p += 1


new_csv.to_csv(csv_path, encoding='UTF-8-SIG', index=False)