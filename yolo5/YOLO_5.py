import torch
import cv2
import numpy as np


# Model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)  # or yolov5m, yolov5l, yolov5x, custom

# Images
img = r'E:\codes_py\Larkimas\Data_source\all_data\0010115706_2.jpg'  # or file, Path, PIL, OpenCV, numpy, list

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
        cv2.imshow('Image', obj['im'])
        cv2.waitKey(0)  # Wait for any key press to close the window
        cv2.destroyAllWindows()  # Close all windows
        print('ok')