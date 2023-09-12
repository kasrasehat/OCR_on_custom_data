import numpy as np
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
import cv2


model_name = 'resnet152'
checkpoint = 'E:/codes_py/Larkimas/checkpoints/resnet152_10_0.9719.pth'
model = torchvision.models.__dict__[model_name](pretrained=False)
classes = torch.load(checkpoint)['classes']
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])