import torch
import os
from torch.utils.data import DataLoader
import random
import math
import matplotlib.pyplot as plt
from skimage import io, color
import cv2
import sys
from colorize_data import ColorizeTestData
from basic_model import Net

def process_command_args(arguments):

    # Specifying the default parameters

    data_dir = './landscape_images/'
    save_dir = './test_result/'
    model_path = './resnet_lab_full_epoch_14.pth'


    for args in arguments:

        if args.startswith("data_dir"):
            data_dir = args.split("=")[1]

        if args.startswith("save_dir"):
            save_dir = args.split("=")[1]

        if args.startswith("model_path"):
            model_path = args.split("=")[1]


    return data_dir, save_dir, model_path

# Processing command arguments

data_dir, save_dir, model_path = process_command_args(sys.argv)

def reconstruct_img(x,colorized):
    l = x[0,:,:,:].cpu().numpy()
    l = np.transpose(l, (1, 2, 0))* 100.0
    ab = colorized.cpu().detach().numpy()
    ab = ab[0,:,:,:].squeeze()
    ab = np.transpose(ab, (1, 2, 0))* 255.0 - 128
    ab[:,:,0] = np.clip(ab[:,:,0], -86, 98)
    ab[:,:,1] = np.clip(ab[:,:,1], -107, 94)
    image = color.lab2rgb(np.concatenate(( l, ab[:,:,:]), axis=2))
    plt.figure()
    plt.imshow(image) 
    plt.show()
    return image

def test_model():

    device = torch.device("cuda")
    #defining the model and loading the weight
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path))

    
    test_list = os.listdir(data_dir)

    test_dataset = ColorizeTestData(test_list)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0,
                            pin_memory=True, drop_last=False)
    with torch.no_grad():
        test_iter = iter(test_dataloader)
        for j in range(len(test_dataloader)):
            torch.cuda.empty_cache()
            x, file_name = next(test_iter)
            x = x.to(device, non_blocking=True, dtype=torch.float)
            colorized = model(x)
            file_name = file_name[0].replace('.jpg','_recons.jpg')

            #generate rgb image from its greyscale input and ab model output 
            img = reconstruct_img(x,colorized)
            #saving the output image in the directory 
            img_bgr = cv2.cvtColor(np.uint8(img*255), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir,file_name), img_bgr)
if __name__ == '__main__':
    test_model()