import torch, torchvision
import torch.nn as nn
import numpy as np
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
import cv2
from Networks import Encoder, DecoderRNN, CustomModel_mse


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

    return resized_image


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


device = torch.device("cuda:0" if True else "cpu")
height, width = 720, 720
trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

YOLO = YOLO("/home/kasra/PycharmProjects/YOLOv8_customize/runs/detect/train/weights/best.pt")
encoder = CustomModel_mse().to(device)
weights = torch.load("/home/kasra/PycharmProjects/Larkimas/model_checkpoints/epoch_4_mse_loss: 0.003.pt")

try:
    encoder.load_state_dict(weights['state_dict encoder'])
except:
    encoder.load_state_dict(weights)

with torch.no_grad():

    image_path = '/home/kasra/kasra_files/data-shenasname/ai_files_20230528_A/0010682961_0.jpg'
    encoder.eval()
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

    draw_rectangle(image_path, transform_vertices([(0, 40), (720, 40), (720, 640), (0, 640)],
                                          get_transformation_matrix(-transforms[0, 8] * 180, transforms[0, 9], transforms[0, -2:])))

    # [(0, 40), (720, 40), (720, 680), (0, 680)]

    theta = -float(transforms[0, 8] * 180)

    # Define the new height and width
    new_height = 720
    new_width = 720

    # Replace 'path_to_your_image.jpg' with the path to your image
    cropped_rotated_resized_image = crop_rotate_resize_pil(image_path, clamped_points, theta, new_height, new_width)
    img = '/home/kasra/PycharmProjects/YOLOv8_customize/extra_files/image1.jpg'
    final_image = cropped_rotated_resized_image[:, :, ::-1]
    save_image(final_image, img)
    results = YOLO.predict(img, save=True, imgsz=640, conf=0.3, save_txt=False, show=True)

    # You can save the result or display it using cv2
    # cv2.imwrite('output_image.jpg', cropped_rotated_resized_image)
    # cv2.imshow('Cropped, Rotated, and Resized Image', cropped_rotated_resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
