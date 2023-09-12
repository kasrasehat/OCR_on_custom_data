import torch
import cv2
import numpy as np
import os
from augment import Augmentor
import cv2


def transform_vertices(vertices, transformation_matrix):
    # Convert the list of tuples to a numpy array of shape (4, 2)
    vertices_array = np.array(vertices)

    # Homogeneous coordinates: Add a column of ones to the vertices array
    vertices_homogeneous = np.hstack([vertices_array, np.ones((vertices_array.shape[0], 1))])

    # Apply the transformation matrix
    transformed_vertices = np.dot(transformation_matrix, vertices_homogeneous.T).T

    # Convert back to Cartesian coordinates
    transformed_vertices_cartesian = transformed_vertices[:, :2] / transformed_vertices[:, 2][:, np.newaxis]
    transformed_vertices_cartesian = transformed_vertices_cartesian + np.ones_like(transformed_vertices_cartesian)*360
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


def draw_rectangle_2vertices(image_path, top_left, bottom_right):
    """
    Draws a rectangle on the image based on given top-left and bottom-right vertices.

    Parameters:
        image_path (str): Path to the image file
        top_left (tuple): (x1, y1) coordinates of the top-left vertex
        bottom_right (tuple): (x2, y2) coordinates of the bottom-right vertex
    """

    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print("Could not open or find the image.")
        return

    # Extract coordinates from the tuples
    x1, y1 = np.int32(top_left[0]), np.int32(top_left[1])
    x2, y2 = np.int32(bottom_right[0]), np.int32(bottom_right[1])

    # Draw the rectangle on the image
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the image
    cv2.imshow('Image with Rectangle', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


# Model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5l').to(device)  # or yolov5m, yolov5l, yolov5x, custom

# Images
img = 'E:/codes_py/Larkimas/Data_source/all_data/classification_data/train/0/0014483084_0.jpg'  # or file, Path, PIL, OpenCV, numpy, list
image = cv2.imread(img)
new_width = 720
new_height = 720
augmentor = Augmentor('E:/codes_py/Larkimas/Data_source/all_data/background')
# Resize the image
image = cv2.resize(image, (new_width, new_height))
# Inference
img = 'E:/codes_py/Larkimas/Data_source/all_data/image1.jpg'
img1 = 'E:/codes_py/Larkimas/Data_source/all_data/image2.jpg'

save_image(image, img)
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
for obj in results.crop():
    if obj['cls'] == 73:
        # Show the image using OpenCV
        cv2.imshow('Image', obj['im'])
        cv2.waitKey(0)  # Wait for any key press to close the window
        cv2.destroyAllWindows()  # Close all windows
        print('ok')

    if obj['cls'] == 0:
        # Show the image using OpenCV
        top_left = (np.int32(obj['box'][0].item()), np.int32(obj['box'][1].item()))
        bottom_right = (np.int32(obj['box'][2].item()), np.int32(obj['box'][3].item()))
        vertices = get_four_vertices(top_left, bottom_right)
        draw_rectangle(img, vertices)
        processed_image, angle, transport, scale = augmentor.scale_rotate_background(img)
        transport1 = [transport[0], transport[1]]
        # transport = np.array(transport) * 720/2
        transform_matrix = get_transformation_matrix(rotation=-angle, scale=scale, translation=transport1)
        save_image(processed_image, img1)
        transformed_vertices = transform_vertices(vertices=vertices, transformation_matrix=transform_matrix)
        # draw_rectangle_2vertices(image_path= img, top_left=top_left, bottom_right= bottom_right)
        draw_rectangle(img1, transformed_vertices)
        # cv2.imshow('Image', obj['im'])
        # cv2.waitKey(0)  # Wait for any key press to close the window
        # cv2.destroyAllWindows()  # Close all windows
        print('ok')

