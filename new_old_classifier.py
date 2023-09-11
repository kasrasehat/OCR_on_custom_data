import torch, torchvision
import torch.nn as nn
import numpy as np
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
import cv2


class detect_new_old():

    def __init__(self):

        self.device_id = [0]
        self.checkpoint = 'E:/codes_py/Larkimas/checkpoints/resnet152_10_0.9719.pth'
        self.model_name = 'resnet152'
        classes = torch.load(self.checkpoint)['classes']
        self.model = torchvision.models.__dict__[self.model_name](pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(classes))
        self.model.cuda()
        self.model.load_state_dict(torch.load(self.checkpoint)['model'])
        self.model.eval()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #
        self.transform = transforms.Compose([
             transforms.ToPILImage(),
             transforms.Resize((720, 720)),
             transforms.ToTensor(),
             normalize,
             self.expand])

    def expand(self, patch):
        patch = torch.unsqueeze(patch, 0)
        return patch.type(torch.float32)

    def soft_max(self, vector):
        vector = np.exp(vector)
        return vector/vector.sum()

    def detect(self, input):
        input = cv2.imread(input)
        input = self.transform(input)
        with torch.no_grad():
            y_pred = self.model(input)

        res = torch.argmax(y_pred)
        return res