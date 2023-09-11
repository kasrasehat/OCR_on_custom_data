from torchvision import transforms
from torch.utils.data import Dataset
import cv2
import os
import pandas as pd
import torch
import numpy as np
import glob
import tqdm
from new_old_classifier import detect_new_old
from augment import Augmentor


def get_four_vertices(top_left, bottom_right):
    """
    Given the coordinates of the top-left and bottom-right vertices of a rectangle,
    returns the coordinates of all four vertices.

    Parameters:
        top_left (tuple): (x1, y1) coordinates of the top-left vertex
        bottom_right (tuple): (x2, y2) coordinates of the bottom-right vertex

    Returns:
        list: List of four vertices in the format [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """

    x1, y1 = top_left
    x2, y2 = bottom_right

    # Top-left, top-right, bottom-right, bottom-left
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def create_csv_if_not_exists(csv_path, existed_csv):
    if not os.path.exists(csv_path):
        # Create an empty DataFrame with the same columns as BB
        df = pd.DataFrame(columns=existed_csv.columns)

        # Save the empty DataFrame to a new CSV file named "AAA.csv"
        df.to_csv(csv_path, index=False)
        print(f"csv file '{csv_path}' created.")
    else:
        print(f"csv file '{csv_path}' already exists.")


def create_folder_if_not_exists(folder_path):
    """
    Create a new folder if it does not already exist.

    Parameters:
    - folder_path (str): The path of the folder to create.

    Returns:
    - None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


def search_by_national_id(national_id, dataframe):
    """
    Search for a given national ID in the dataframe national id column

    Parameters:
        national_id (int or str): The national ID to search for.
        dataframe (pd.DataFrame): The DataFrame to search in.

    Returns:
        True or False
    """
    # Search for the national ID in the 'NATIONAL_ID' column
    match = dataframe[dataframe['NATIONAL_ID'] == int(national_id)]

    # If a match is found
    if not match.empty:
        return [True, match.iloc[0].name]
    else:
        return False


def save_image(image, path):
    """
    Save an image to a specified path.

    Parameters:
        image (ndarray): The image to be saved.
        path (str): The path where the image will be saved.
    """
    # Create the directory if it doesn't exist
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Save the image
    cv2.imwrite(path, image)


augmentor = Augmentor('E:/codes_py/Larkimas/Data_source/all_data/background')
detector = detect_new_old()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5l').to(device)
files = glob.glob('E:/codes_py/Larkimas/Data_source/UBUNTU 20_0/*.CSV') + glob.glob('E:/codes_py/Larkimas/Data_source/UBUNTU 20_0/*.csv')
file_list = []

# create list of files including existed images and corresponding excel file
for file in files:
    file_list.append([file.replace('\\', '/'), file.split('.')[0].replace('metadata', 'files').replace('\\', '/')])

# read image file and its corresponding Excel file and create new ones for augmented data
for index, file in tqdm.tqdm(enumerate(file_list)):

    # create new folder for augmented images
    folder_path = file[1] + '_A'
    create_folder_if_not_exists(folder_path)
    labels = pd.read_csv(file[0])
    labels[['CLASS', 'PERSON_COORD', 'ROTATION', 'SCALE']] = None
    labels.to_csv(file[0], index=False)
    csv_path = file[0].split('.')[0] + '_A' + '.' + file[0].split('.')[1]
    create_csv_if_not_exists(csv_path, labels)
    new_csv = pd.read_csv(csv_path)
    tot_images = 0
    # read images one by one from each folder and its corresponding Excel file
    for idx, file_name in enumerate(os.listdir(file[1])):

        national_id, flag = file_name.split('.')[0].split('_')
        exist = search_by_national_id(national_id, labels)

        if isinstance(exist, list):
            if int(flag) in [0, 1]:
                labels.loc[exist[1], 'CLASS'] = int(flag)
            else:
                labels.loc[exist[1], 'CLASS'] = detector.detect(file[1] + '/' + file_name)

            image = cv2.imread(file[1] + '/' + file_name)
            new_width = 720
            new_height = 720
            if flag != 1:
                # Resize the image
                image = cv2.resize(image, (new_width, new_height))
                # Inference
                save_path = 'E:/codes_py/Larkimas/Data_source/all_data/image.jpg'
                save_image(image, save_path)
                results = model(save_path)
                for obj in results.crop():
                    if obj['cls'] == 0:
                        tot_images += 1
                        top_left = (np.int32(obj['box'][0].item()), np.int32(obj['box'][1].item()))
                        bottom_right = (np.int32(obj['box'][2].item()), np.int32(obj['box'][3].item()))
                        vertices = get_four_vertices(top_left, bottom_right)
                        labels.loc[exist[1], 'PERSON_COORD'] = vertices
                        new_csv.loc[len(new_csv)] = labels.loc[exist[1]][:8]




