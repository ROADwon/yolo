import torch
import cv2 as cv
import pandas as ps
import numpy as np
import os
import sys

from ultralytics import YOLO
from PIL import Image
from glob import glob

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Model :
    def __init__(self, device) :
        self.model = YOLO('yolov8n.pt')
        self.device = device
        self.model = self.model.to(self.device)

    def data_loader(self, path):
        file_path = os.path.join(path, 'images')
        images = glob(file_path + '/*.png')

