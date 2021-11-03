from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import os
from scipy import misc
import imageio

np.random.seed(0)
torch.manual_seed(0)

class ColorizeData(Dataset):
    def __init__(self,files_name):  
        # Initialize dataset, you may use a second dataset for validation if required
        # Use the input transform to convert images to grayscale
        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256,256)),
                                          T.Grayscale()
                                        #   T.Normalize((0.5), (0.5))
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.ToTensor(),
                                           T.Resize(size=(256,256))])
                                        #    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                        #    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.data_dir = '/content/gdrive/My Drive/image_colorization/landscape_images'
        self.files_name = files_name 
        self.dataset_size = len(self.files_name)
        print("data size = ",self.dataset_size)
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, index):
        image = np.asarray(io.imread(os.path.join(self.data_dir, self.files_name[index])))
        lab = color.rgb2lab(image)
        lab = (lab + 128) / 255.0
        ab = lab[:,:,1:3]
        # Return the input tensor and output tensor for training
        return self.input_transform(image/ 255.0), self.target_transform(ab)

class ColorizeTestData(Dataset):
    def __init__(self,files_name):  
        # Initialize dataset, you may use a second dataset for validation if required
        # Use the input transform to convert images to grayscale
        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256,256)),
                                          T.Grayscale()
                                          ])
        # Use this on target images(colorful ones)
        self.data_dir = '/content/gdrive/My Drive/image_colorization/landscape_images'
        self.files_name = files_name 
        self.dataset_size = len(self.files_name)
        print("data size = ",self.dataset_size)
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, index):
        image = np.asarray(io.imread(os.path.join(self.data_dir, self.files_name[index])))
        # Return the input tensor and output tensor for training
        return self.input_transform(image/ 255.0), self.files_name[index]