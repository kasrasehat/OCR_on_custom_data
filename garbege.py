# import torchvision
# from torchvision import transforms
# from PIL import ImageFile
# import cv2
# import numpy as np
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# transforms.Compose([
#     transforms.Resize((256, 256)),
#     # transforms.RandomResizedCrop(224),
#     transforms.RandomCrop(224),
#     transforms.RandomRotation(30),
#     transforms.RandomGrayscale(p=0.4),
#     # transforms.Grayscale(num_output_channels=3),
#     transforms.RandomAffine(45, shear=0.2),
#     # transforms.ColorJitter(),
#     transforms.RandomHorizontalFlip(),
#     # transforms.Lambda(utils.randomColor),
#     # transforms.Lambda(utils.randomBlur),
#     # transforms.Lambda(utils.randomGaussian),
#     transforms.ToTensor(),
#     normalize ,])
#
#
# # Load the image
# image = cv2.imread('Data_source/all_data/rangi/0013687239_0.jpg')
#
# # Display the original image
# cv2.imshow('Original Image', image)
# cv2.waitKey(0)
#
# # Convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Convert grayscale image back to BGR format
# gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
#
# # Display the grayscale BGR image
# cv2.imshow('Grayscale BGR Image', gray_bgr)
# cv2.waitKey(0)
#
# # Cleanup windows
# cv2.destroyAllWindows()

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from new_old_classifier import detect_new_old
# Load the image
# path = 'E:/codes_py/Larkimas/Data/train/0/0010327142_2.jpg'
# detector = detect_new_old()
# a = detector.detect(path)
import pandas as pd
a = pd.read_csv('/home/kasra/kasra_files/data-shenasname/data_loc')