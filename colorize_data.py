from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import os
from scipy import misc
import imageio

class ColorizeData(Dataset):
    def __init__(self):  
        # Initialize dataset, you may use a second dataset for validation if required
        # Use the input transform to convert images to grayscale
        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256,256)),
                                          T.Grayscale(),
                                          T.Normalize((0.5), (0.5))
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.ToTensor(),
                                           T.Resize(size=(256,256)),
                                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.data_dir = '/content/gdrive/My Drive/image_colorization/landscape_images'
        self.files_name = os.listdir(self.data_dir)
        self.dataset_size = len(self.files_name)
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, index):
        image = np.asarray(imageio.imread(os.path.join(self.data_dir, self.files_name[idx])))
        # Return the input tensor and output tensor for training
        return self.input_transform(image), self.target_transform(image)