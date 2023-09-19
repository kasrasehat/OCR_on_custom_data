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


def draw_rectangle(image_path, vertices):
    """
    Draws a rectangle on the image based on given vertices.

    Parameters:
        image_path (str): Path to the image file
        vertices (list): List of four vertices in the format [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """

    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print("Could not open or find the image.")
        return

    # Convert vertices to a NumPy array
    points = np.array(vertices, np.int32)
    points = points.reshape((-1, 1, 2))

    # Draw the rectangle on the image
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=3)
    #image = cv2.resize(image, (1920, 1080))
    # Show the image
    cv2.imshow('Image with Rectangle', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def transform_vertices(vertices, transformation_matrix):
    # Convert the list of tuples to a numpy array of shape (4, 2)
    vertices_array = np.array(vertices) - np.ones_like(vertices)*360

    # Homogeneous coordinates: Add a column of ones to the vertices array
    vertices_homogeneous = np.hstack([vertices_array, np.ones((vertices_array.shape[0], 1))])

    # Apply the transformation matrix
    transformed_vertices = np.dot(transformation_matrix, vertices_homogeneous.T).T

    # Convert back to Cartesian coordinates
    transformed_vertices_cartesian = transformed_vertices[:, :2] / transformed_vertices[:, 2][:, np.newaxis]
    transformed_vertices_cartesian = transformed_vertices_cartesian + np.ones_like(transformed_vertices_cartesian)*360
    for i in range(transformed_vertices_cartesian.shape[0]):
        for j in range(transformed_vertices_cartesian.shape[1]):
            transformed_vertices_cartesian[i, j] = int(transformed_vertices_cartesian[i, j])
    # Convert the numpy array back to a list of tuples
    transformed_vertices_list = [tuple(coord) for coord in transformed_vertices_cartesian]

    return transformed_vertices_list


def get_transformation_matrix(rotation, scale, translation):
    """
    Calculate the transformation matrix based on the given rotation, scale, and translation.

    Parameters:
        rotation (float): The rotation angle in degrees.
        scale (float): The scale factor.
        translation (tuple): A tuple (tx, ty) representing the x and y translation.

    Returns:
        np.array: The 3x3 transformation matrix.
    """
    # Convert rotation angle to radians
    theta = np.radians(rotation)

    # Create individual transformation matrices
    # Rotation matrix
    T_rot = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0            ,              0, 1]
    ])

    # Scale matrix
    T_scale = np.array([
        [scale,    0, 0],
        [0,    scale, 0],
        [0,        0, 1]
    ])

    # Translation matrix
    tx, ty = translation
    T_trans = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0,  1]
    ])

    # Combine transformations
    T = np.dot(T_trans, np.dot(T_rot, T_scale))

    return T


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
        df.to_csv(csv_path, encoding='UTF-8-SIG', index=False)
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


def save_image(image, path, overwrite=True):
    """
    Save an image to a specified path.

    Parameters:
        image (ndarray): The image to be saved.
        path (str): The path where the image will be saved.
        overwrite (bool): Whether to overwrite the file if it already exists.
    """
    # Create the directory if it doesn't exist
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Check if an image with the same path already exists
    if os.path.exists(path):
        if overwrite:
            cv2.imwrite(path, image)
    else:
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
    augmented_image_folder_path = file[1] + '_A'
    create_folder_if_not_exists(augmented_image_folder_path)
    labels = pd.read_csv(file[0], encoding='UTF-8-SIG')
    labels[['CLASS', 'PERSON_COORD', 'ROTATION', 'SCALE', 'TRANSPORT']] = None
    labels.to_csv(file[0], encoding='UTF-8-SIG', index=False)
    csv_path = file[0].split('.')[0] + '_A' + '.' + file[0].split('.')[1]
    create_csv_if_not_exists(csv_path, labels)
    new_csv = pd.read_csv(csv_path, encoding='UTF-8-SIG')
    tot_images = 0
    # read images one by one from each folder and its corresponding Excel file
    for idx, file_name in enumerate(os.listdir(file[1])):
        try:
            national_id, flag = file_name.split('.')[0].split('_')
            flag = int(flag)
            exist = search_by_national_id(national_id, labels)

            if isinstance(exist, list):
                if int(flag) in [0, 1]:
                    card_type = int(flag)
                else:
                    card_type = detector.detect(file[1] + '/' + file_name)

                p = 0
                if pd.isna(labels.loc[exist[1], 'CLASS']):
                    labels.loc[exist[1], 'CLASS'] = card_type
                    labels.loc[exist[1], 'ROTATION'] = 0
                    labels.loc[exist[1], 'SCALE'] = 0
                    labels.at[exist[1], 'TRANSPORT'] = [0, 0]
                    p = exist[1]
                elif labels.loc[exist[1], 'CLASS'] != card_type:
                    l1 = len(labels)
                    labels.loc[l1, labels.columns[:8]] = labels.loc[exist[1], labels.columns[:8]]
                    labels.loc[l1, 'CLASS'] = card_type
                    labels.loc[l1, 'ROTATION'] = 0
                    labels.loc[l1, 'SCALE'] = 0
                    labels.at[l1, 'TRANSPORT'] = [0, 0]
                    p = l1

                image = cv2.imread(file[1] + '/' + file_name)
                new_width = 720
                new_height = 720
                # Resize the image
                image = cv2.resize(image, (new_width, new_height))
                if flag != 1:
                    # Inference
                    save_path = 'E:/codes_py/Larkimas/Data_source/all_data/image.jpg'
                    save_image(image, save_path, overwrite=True)
                    results = model(save_path)
                    for obj in results.crop():
                        if obj['cls'] == 0:
                            tot_images += 1
                            top_left = (np.int32(obj['box'][0].item()), np.int32(obj['box'][1].item()))
                            bottom_right = (np.int32(obj['box'][2].item()), np.int32(obj['box'][3].item()))
                            vertices = get_four_vertices(top_left, bottom_right)
                            labels.at[p, 'PERSON_COORD'] = vertices
                            processed_image, angle, transport, scale = augmentor.scale_rotate_background(
                                file[1] + '/' + file_name)
                            transport1 = [transport[0], transport[1]]
                            transform_matrix = get_transformation_matrix(rotation=-angle, scale=scale,
                                                                         translation=transport1)
                            transformed_vertices = transform_vertices(vertices=vertices,
                                                                      transformation_matrix=transform_matrix)
                            processed_image_path = augmented_image_folder_path + '/' + file_name
                            if not os.path.exists(processed_image_path):
                                l2 = len(new_csv)
                                new_csv.loc[l2, new_csv.columns[:8]] = labels.loc[p, labels.columns[:8]]
                                new_csv.at[l2, 'PERSON_COORD'] = transformed_vertices
                                new_csv.loc[l2, 'ROTATION'] = np.round(np.radians(angle) / np.pi, 2)
                                new_csv.loc[l2, 'SCALE'] = scale
                                new_csv.at[l2, 'TRANSPORT'] = transport1
                            save_image(processed_image, augmented_image_folder_path + '/' + file_name)
                            if tot_images % 100 == 0:
                                new_csv.to_csv(csv_path, encoding='UTF-8-SIG', index=False)
                                labels.to_csv(file[0], encoding='UTF-8-SIG', index=False)
                            # draw_rectangle(file[1] + '_A' + '/' + file_name, transformed_vertices)
                else:
                    tot_images += 1
                    top_left = (np.int32(0), np.int32(0))
                    bottom_right = (np.int32(0), np.int32(0))
                    vertices = get_four_vertices(top_left, bottom_right)
                    labels.at[p, 'PERSON_COORD'] = vertices
                    processed_image, angle, transport, scale = augmentor.scale_rotate_background(
                        file[1] + '/' + file_name)
                    transport1 = [transport[0], transport[1]]
                    # transport = np.array(transport) * 720/2
                    transform_matrix = get_transformation_matrix(rotation=-angle, scale=scale,
                                                                 translation=transport1)
                    transformed_vertices = transform_vertices(vertices=vertices,
                                                              transformation_matrix=transform_matrix)
                    processed_image_path = augmented_image_folder_path + '/' + file_name
                    if not os.path.exists(processed_image_path):
                        l2 = len(new_csv)
                        new_csv.loc[l2, new_csv.columns[:8]] = labels.loc[p, labels.columns[:8]]
                        new_csv.at[l2, 'PERSON_COORD'] = vertices
                        new_csv.loc[l2, 'ROTATION'] = np.round(np.radians(angle) / np.pi, 2)
                        new_csv.loc[l2, 'SCALE'] = scale
                        new_csv.at[l2, 'TRANSPORT'] = transport1

                    save_image(processed_image, augmented_image_folder_path + '/' + file_name)
                    if tot_images % 100 == 0:
                        new_csv.to_csv(csv_path, encoding='UTF-8-SIG', index=False)
                        labels.to_csv(file[0], encoding='UTF-8-SIG', index=False)
        except Exception as e:
            print(f'{file_name} in the file of {file[1]}  is corrupted.  Error: {e}')

    print(f'total number of images augmented from {file[1]} is {tot_images}')




