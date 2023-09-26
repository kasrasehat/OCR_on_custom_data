import pandas as pd
import torch
import cv2
import numpy as np
import os
from augment import Augmentor
import cv2
from ast import literal_eval


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


img = 'E:/codes_py/Larkimas/Data_source/UBUNTU 20_0/ai_files_20230606_2_A/0011852879_0.jpg'
csv_path = '/'.join(str(f) for f in img.split('.')[0].split('/')[:-1]).replace('files', 'metadata') + '.csv'
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



