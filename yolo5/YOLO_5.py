import torch
import cv2
import numpy as np
import os


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
model = torch.hub.load('ultralytics/yolov5', 'yolov5n').to(device)  # or yolov5m, yolov5l, yolov5x, custom

# Images
img = 'E:/codes_py/Larkimas/Data_source/all_data/classification_data/train/3/0018465811_2.jpg'  # or file, Path, PIL, OpenCV, numpy, list
image = cv2.imread(img)
new_width = 224
new_height = 224

# Resize the image
image = cv2.resize(image, (new_width, new_height))
# Inference
img = 'E:/codes_py/Larkimas/Data_source/all_data/image.jpg'
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
        # draw_rectangle_2vertices(image_path= img, top_left=top_left, bottom_right= bottom_right)
        draw_rectangle(img, vertices)
        # cv2.imshow('Image', obj['im'])
        # cv2.waitKey(0)  # Wait for any key press to close the window
        # cv2.destroyAllWindows()  # Close all windows
        print('ok')

