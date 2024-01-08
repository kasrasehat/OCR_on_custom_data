import torch, torchvision
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
import cv2
from Networks import Encoder, DecoderRNN, CustomModel_mse
from PIL import Image
from ultralytics import YOLO
import os
from custom_dataloader import cropped_DataLoader, All_ID_card_DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader


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

def crop_rotate_resize_pil(image_path, points, theta, new_height, new_width):
    # Load the image
    image = Image.open(image_path)
    image = image.resize((new_width, new_height))

    # Define the rectangle points and crop the image
    # PIL expects the cropping box in the format: (left, upper, right, lower)
    left, top, right, bottom = cv2.boundingRect(np.array(points, dtype=np.int32))
    cropped_image = image.crop((left, top, left + right, top + bottom))

    # Rotate the image
    # PIL rotates counter-clockwise, hence we use negative value for theta
    rotated_image = cropped_image.rotate(theta, expand=True)

    # Resize the image
    resized_image = np.array(rotated_image.resize((new_width, new_height)))

    return resized_image


def crop_rotate_resize(image_path, points, theta, new_height, new_width):
    # Load the image
    image = cv2.imread(image_path)

    # Define the rectangle points
    rect = cv2.boundingRect(np.array([points], dtype=np.int32).reshape((-1,1,2)))

    # Crop the image using the rectangle points
    cropped_image = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

    # Rotate the image
    # Calculate the center of the cropped image
    center = (rect[2] // 2, rect[3] // 2)

    # Define the rotation matrix
    M = cv2.getRotationMatrix2D(center, theta, 1)

    # Perform the rotation
    rotated_image = cv2.warpAffine(cropped_image, M, (rect[2], rect[3]))

    # Resize the image
    resized_image = cv2.resize(rotated_image, (new_width, new_height))

    return rotated_image


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



def draw_rectangle(image_path, vertices):
    """
    Draws a rectangle on the image based on given vertices.

    Parameters:
        image_path (str): Path to the image file
        vertices (list): List of four vertices in the format [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """

    # Read the image using OpenCV
    image = cv2.imread(image_path)
    image = cv2.resize(image, (720, 720))

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

def expand(patch):
    patch = torch.unsqueeze(patch, 0)
    return patch.type(torch.float32)

def clamp_points(points, min_val=0, max_val=720):
    clamped_points = []
    for x, y in points:
        clamped_x = min(max(x, min_val), max_val)
        clamped_y = min(max(y, min_val), max_val)
        clamped_points.append((clamped_x, clamped_y))
    return clamped_points

def process_tensor(tensor, image, file_name, target, kk):
    for i in range(tensor.size(0)):
        # Check if the last element (label) is 11
        YOLO_name = names[tensor[i, -1].item()]
        if tensor[i, -1] == 11:
            # Extract coordinates
            x1, y1, x2, y2 = map(int, tensor[i, :4])
            if x1  > 0:
                x1 = x1
            else:
                x1 = 0

            if x2 +10 > 0:
                x2 = x2 +10
            else:
                x2 = 0

            if y1 -20 > 0:
                y1 = y1 -20
            else:
                y1 = 0

            if y2 + 10 > 0:
                y2 = y2 + 10
            else:
                y2 = 0

            # Crop and rotate the bounding box
            cropped_image = image[y1:y2, x1:x2]
            cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
            cropped_image = cv2.resize(cropped_image, (256, 128))
            save_path = f'/home/kasra/kasra_files/data-shenasname/cropped_data1/{kk}' + '/'+ ','.join(file_name.split('/')[-2:]) + ','+ YOLO_name+ ','+ target[YOLO_name][0]

            # Save the image
            cv2.imwrite(save_path, cropped_image)
        elif tensor[i, -1] in [0, 2, 5, 6, 9, 10]:
            continue
        else:
            # Extract coordinates
            x1, y1, x2, y2 = map(int, tensor[i, :4])

            # Crop and rotate the bounding box
            cropped_image = image[y1:y2, x1:x2]
            cropped_image = cv2.resize(cropped_image, (256, 128))

            save_path =f'/home/kasra/kasra_files/data-shenasname/cropped_data1/{kk}' + '/'+ ','.join(file_name.split('/')[-2:]) + ','+ YOLO_name+ ','+ target[YOLO_name][0]
            # Save the image
            cv2.imwrite(save_path, cropped_image)

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



device = torch.device("cuda:0" if True else "cpu")
height, width = 720, 720
trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

YOLO = YOLO("/home/kasra/PycharmProjects/YOLOv8_customize/runs/detect/train/weights/best.pt")  # load a pretrained model (recommended for training)
encoder = CustomModel_mse().to(device)
weights = torch.load("/home/kasra/PycharmProjects/Larkimas/model_checkpoints/epoch_4_mse_loss: 0.003.pt")
try:
    encoder.load_state_dict(weights['state_dict encoder'])
except:
    encoder.load_state_dict(weights)

names = {0: 'back_national_card',
         1: 'Persian Birth Date',
         2: 'expiration_date',
         3: 'Last Name',
         4: 'Father Name',
         5: 'front_national_card',
         6: 'image',
         7: 'First Name',
         8: 'National_ID',
         9: 'new_birth_certificate',
        10: 'old_birth_certificate',
        11: 'National ID Serial'}

dataset1 = cropped_DataLoader(data_file='/home/kasra/kasra_files/data-shenasname/just_id_card', transform=trans)
dataloader1 = DataLoader(dataset1, batch_size=1, shuffle=True, drop_last=True)
    # You can save the result or display it using cv2
    # cv2.imwrite('output_image.jpg', cropped_rotated_resized_image)
    # cv2.imshow('Cropped, Rotated, and Resized Image', cropped_rotated_resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
p = 0
tot_time = 0
with torch.no_grad():
    encoder.eval()

    for batch_idx, (input_path, target) in enumerate(dataloader1):
        if batch_idx % 10000 == 0:
            kk = str(batch_idx)
            create_folder_if_not_exists('/home/kasra/kasra_files/data-shenasname/cropped_data2/'+ kk)
        try:
            image_path = input_path[0]
            target = target
            image = cv2.imread(image_path)
            image = trans(image).to(device)
            input = expand(image)
            transforms = encoder(input)
            transforms[0, 0:8] = transforms[0, 0:8] * 720
            transforms[0, 8] = (transforms[0, 8] * 2) -1
            if transforms[0, 9]>1:
                transforms[0, 9] = 1
            elif transforms[0, 9] < 0:
                transforms[0, 9] = 0

            transforms[0, -2:] =(((transforms[0, -2:] * 2) - torch.ones_like(transforms[0, -2:])) *360).reshape(1, 2)
            transforms = transforms.cpu()
            print(transforms)
            points = transform_vertices([(0, 0), (720, 0), (720, 720), (0, 720)],
                                                   get_transformation_matrix(-transforms[0, 8] * 180, transforms[0, 9], transforms[0, -2:]))
            clamped_points = clamp_points(points)

            # draw_rectangle(image_path, transform_vertices([(0, 40), (720, 40), (720, 640), (0, 640)],
            #                                       get_transformation_matrix(-transforms[0, 8] * 180, transforms[0, 9], transforms[0, -2:])))

            # [(0, 40), (720, 40), (720, 680), (0, 680)]

            theta = -float(transforms[0, 8] * 180)

            # Define the new height and width
            new_height = 720
            new_width = 720

            # Replace 'path_to_your_image.jpg' with the path to your image
            cropped_rotated_resized_image = crop_rotate_resize_pil(image_path, clamped_points, theta, new_height, new_width)
            final_image = cropped_rotated_resized_image # [:, :, ::-1]
            img = '/home/kasra/PycharmProjects/YOLOv8_customize/extra_files/image1.jpg'

            save_image(final_image, img)
            results = YOLO.predict(img, save=False, imgsz=640, conf=0.3, save_txt=False, show=False)
            process_tensor(results[0].boxes.data, cropped_rotated_resized_image,
                                  file_name = image_path, target=target, kk= kk)  # returns xyxy of bounding box + confidence and class number
            tot_time = results[0].speed['preprocess']+ results[0].speed['inference']+ results[0].speed['postprocess']+ tot_time
            p += 1
            print(f'progress is {batch_idx*100/len(dataloader1)}%')
        except Exception as e:
            print(f"An error occurred while reading : {e}\n")

print(tot_time/p)
