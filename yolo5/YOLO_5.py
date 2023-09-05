import torch
import cv2
import numpy as np


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

    # Show the image
    cv2.imshow('Image with Rectangle', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)  # or yolov5m, yolov5l, yolov5x, custom

# Images
img = r'/home/kasra/kasra_files/data-shenasname/ai_files_20230606/0011926661_2.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
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
        top_left = (obj['box'][0].item(), obj['box'][1].item())
        bottom_right = (obj['box'][2].item(), obj['box'][3].item())
        vertices = get_four_vertices(top_left, bottom_right)
        draw_rectangle(img, vertices)
        cv2.imshow('Image', obj['im'])
        cv2.waitKey(0)  # Wait for any key press to close the window
        cv2.destroyAllWindows()  # Close all windows
        print('ok')

