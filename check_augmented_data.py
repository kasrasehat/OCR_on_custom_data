import pandas as pd
import torch
import cv2
import numpy as np
import os
from augment import Augmentor
import cv2
from ast import literal_eval


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


img = '/home/kasra/kasra_files/data-shenasname/ai_files_20230606_2_A/0011852879_0.jpg'
csv_path = '/'.join(str(f) for f in img.split('.')[0].split('/')[:-2]) + '/'+ '/'.join(str(f) for f in img.split('.')[0].split('/')[-2:-1]).replace('files', 'metadata') + '.csv'
id, cls = img.split('.')[0].split('/')[-1].split('_')
id = int(id)
labels = pd.read_csv(csv_path, encoding='UTF-8-SIG', converters={'feature': literal_eval})
# Find the row where 'NATIONAL_ID' is equal to 'id' and 'CLASS' is equal to 'cls'
matching_row_list = labels[(labels['NATIONAL_ID'] == int(id)) & (labels['CLASS'] == int(cls))].index.tolist()
if len(matching_row_list) == 0 and int(cls) in [2, 3]:
    if int(cls) == 2:
        cls = 3
        matching_row = labels[(labels['NATIONAL_ID'] == int(id)) & (labels['CLASS'] == int(cls))].index.tolist()[0]
    elif int(cls) == 3:
        cls = 2
        matching_row = labels[(labels['NATIONAL_ID'] == int(id)) & (labels['CLASS'] == int(cls))].index.tolist()[0]
else:
    matching_row = matching_row_list[0]
vertices = literal_eval(labels.iloc[matching_row]['PERSON_COORD'])
draw_rectangle(img, vertices)
draw_rectangle(img, transform_vertices([(0,40),(720, 40),(720,680),(0, 680)], get_transformation_matrix(-labels.iloc[matching_row].ROTATION*180, labels.iloc[matching_row].SCALE, literal_eval(labels.iloc[matching_row].TRANSPORT))))



