from custom_dataloader import All_ID_card_DataLoader
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
        df = pd.DataFrame(columns=['location'])

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


files = glob.glob('/home/kasra/kasra_files/data-shenasname/*.CSV') + glob.glob('/home/kasra/kasra_files/data-shenasname/*.csv')
file_list = []
for file in files:
    file_list.append([file, file.split('.')[0].replace('metadata', 'files')])
tokens = []
max = 0
min = 100
p = 0
dataset1 = All_ID_card_DataLoader(data_file='/home/kasra/kasra_files/data-shenasname/data_loc', transform=trans)
dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True, drop_last=True)

# Iterate through the DataLoader to get one dictionary for each data point
for batch in dataloader1:
    images, labels = batch  # Unpack the batch into images and labels
    # for i in range(len(images)):
    #     image = images[i]
    #     label = labels['passage'][i]
        # Process the image and label as needed
        # 'image' and 'label' are now one dictionary for each data point in the batch
#     for i, data in tqdm.tqdm(enumerate(dataloader1)):
#         print(i)
#         print(data[0].shape)
#         for p in range(16):
#             if len(list(data[1])[p]) > max:
#                 max = len(list(data[1])[p])
#
#         for j in range(16):
#             if len(list(data[1])[j]) < min:
#                 min = len(list(data[1])[j])
#
#         for k in range(len(list(data[1]))):
#             for token in list(list(data[1])[k]):
#                 if token not in tokens:
#                     tokens.append(token)
#
#
#
# print(max)
# print(min)
# with open("tokens_list.pkl", "wb") as f:
#     pickle.dump(tokens, f)
