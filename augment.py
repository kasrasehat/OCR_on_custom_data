import random
from PIL import Image
import os
import numpy as np
import cv2
import torchvision.transforms as transforms


class Augmentor:
    def __init__(self, backgrounds_path: str = None, output_size: tuple = (720, 720)):
        self.backgrounds_path = backgrounds_path
        self.backgrounds = os.listdir(self.backgrounds_path)
        self.output_size = output_size

    def scale_rotate_background(self, image_path):
        # Load the image
        image = Image.open(image_path)

        # Randomly scale the image
        scale = random.uniform(0.2, 0.6)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)

        # Rotate the image
        angle = random.uniform(-180, 180)
        image = image.rotate(angle, expand=True)
        image = np.array(image)
        new_size = (int(image.shape[1]), int(image.shape[0]))

        # Load a random background image
        background_image_path = os.path.join(self.backgrounds_path, random.choice(self.backgrounds))
        background_image = cv2.imread(background_image_path)

        # Resize the background to be larger than the image
        coef = np.random.randint(10, 26)/10
        background_image = cv2.resize(background_image, (int(coef*new_size[0]), int(coef*new_size[1])), interpolation=cv2.INTER_NEAREST)

        # Define a region of interest (ROI) where you want to place the image on the background
        x_offset1 = int((background_image.shape[1] - image.shape[1]) / 2)
        y_offset1 = int((background_image.shape[0] - image.shape[0]) / 2)
        x_offset = random.randint(0, 2 * x_offset1)
        y_offset = random.randint(0, 2 * y_offset1)
        if coef == 1:
            transport = np.array([0, 0])
        else:
            # transport = [(x_offset - x_offset1) / x_offset1, (y_offset - y_offset1) / y_offset1]
            transport = [x_offset - x_offset1, y_offset - y_offset1]

        roi = background_image[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]]

        # Ensure the image sizes match the ROI
        image = cv2.resize(image, (roi.shape[1], roi.shape[0]))

        # Use the bitwise AND operation to create a mask from the image
        mask_inv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask_inv = cv2.threshold(mask_inv, 1, 255, cv2.THRESH_BINARY_INV)

        # Bitwise-AND mask and ROI
        background_roi = cv2.bitwise_and(roi[:, :, ::-1], roi[:, :, ::-1], mask=mask_inv)

        # Bitwise-OR mask with image
        image = cv2.bitwise_or(image, background_roi)

        # Modify the ROI of the background image with the new image
        background_image[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image[:, :, ::-1]
        resized_image = cv2.resize(background_image, self.output_size)
        transport_ratio = [self.output_size[1] / background_image.shape[1], self.output_size[0] / background_image.shape[0]]
        transport = [np.round(transport_ratio[0] * transport[0], 1), np.round(transport_ratio[1] * transport[1], 1)]
        return resized_image, np.round(angle, 2), transport, np.round(1/coef, 2)

    def _3ch_grayscale(self, image_path):
        # Load the image
        image = Image.open(image_path)

        # Define the transform to convert the image to grayscale
        transform = transforms.Grayscale(num_output_channels=3)

        # Apply the transform
        gray_image = transform(image)

        # Convert to numpy array for visualization
        numpy_image = np.array(gray_image)
        resized_image = cv2.resize(numpy_image, self.output_size)
        return resized_image

    def _1ch_grayscale(self, image_path):
        # Load the image
        image = Image.open(image_path)

        # Convert the image to grayscale with 1 channel
        gray_image_1ch = image.convert('L')

        # Convert to numpy array for visualization
        numpy_image_1ch = np.array(gray_image_1ch)
        resized_image = cv2.resize(numpy_image_1ch, self.output_size)

        return resized_image
