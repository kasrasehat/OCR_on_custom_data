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


device = torch.device("cuda:0" if True else "cpu")
height, width = 720, 720
trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

encoder = CustomModel_mse().to(device)
weights = torch.load("/home/kasra/PycharmProjects/Larkimas/model_checkpoints/epoch_4_mse_loss: 0.003.pt")

try:
    encoder.load_state_dict(weights['state_dict encoder'])
except:
    encoder.load_state_dict(weights)

with torch.no_grad():

    image_path = '/home/kasra/kasra_files/data-shenasname/ai_files_20230706_1_A/0010073851_0.jpg'
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
    # draw_rectangle(image_path, transforms[0, 0:8])
    draw_rectangle(image_path, transform_vertices([(0, 40), (720, 40), (720, 680), (0, 680)],
                                           get_transformation_matrix(-transforms[0, 8] * 180, transforms[0, 9], transforms[0, -2:])))




