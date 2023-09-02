# import random
# from PIL import Image
# import os
# # List of background images
# backgrounds = os.listdir('Data_source/all_data/background')
#
# # Load the image
# image = Image.open('Data_source/all_data/0010011692_0.jpg')
#
# # Display the original image
# image.show()
#
# # Randomly scale the image
# scale = random.uniform(0.5, 1.5)
# new_size = (int(image.width * scale), int(image.height * scale))
# image = image.resize(new_size, Image.ANTIALIAS)
#
# # Rotate the image
# angle = random.uniform(-90, 90)
# image = image.rotate(angle, expand=True)

# # Add a random background
# background_image = Image.open('Data_source/all_data/background/' + str(random.choice(backgrounds)))
# # Add a random background
#
#
# # Resize background to match the object image size
# background_image = background_image.resize(image.size, Image.ANTIALIAS)
#
# # Ensure the image has an alpha channel
# image = image.convert('RGBA')
#
# # Convert white (also shades of whites)
# # pixels to transparent
# r, g, b, alpha = image.split()
# white = Image.merge('RGB', (r, g, b))
# white = white.convert('L')
# mask = Image.new('L', image.size, 255)
# mask = mask.convert('L')
# mask = Image.composite(mask, white, mask)
# image.putalpha(mask)
#
# # Combine the images
# combined_image = Image.alpha_composite(background_image, image)
#
# # Show the transformed image with new background
# combined_image.show()
import random
import PIL
from PIL import Image
import os
import numpy as np
import cv2
import numpy as np
import random
import os
# Load the image
image = Image.open('E:/codes_py/Larkimas/Data_source/all_data/new-shen/0010115706_2.jpg')
# Display the original image
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
backgrounds = os.listdir('E:/codes_py/Larkimas/Data_source/all_data/background')
background_image = cv2.imread('E:/codes_py/Larkimas/Data_source/all_data/background'+'/' + str(random.choice(backgrounds)))

# Resize the background to be larger than the image
coef = np.random.randint(11, 18)/10
background_image = cv2.resize(background_image, (int(coef*new_size[0]), int(coef*new_size[1])), interpolation=cv2.INTER_NEAREST)

# Define a region of interest (ROI) where you want to place the image on the background
# Here it is defined such that the image will be centered on the background
x_offset = int((background_image.shape[1] - image.shape[1]) / 2)
y_offset = int((background_image.shape[0] - image.shape[0]) / 2)
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
resized_image = cv2.resize(background_image, (1080, 920))

# Display the image
cv2.imshow('Image with Background', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
