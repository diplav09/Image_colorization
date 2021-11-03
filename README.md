# Image_colorization

#### 1. File contents

1. train.py contains train code 
2. test.py contains inference code
3. colorize_data.py contains dataloader
4. basic_model.py  contains the model architecture
5. msssim.py contains the code for MSSIM loss function used in training code
6. models/ contain the best epoch weight file
7. data/ visualization results

#### 2. Prerequisites

- Python: scipy, numpy, imageio and pillow packages
- [PyTorch + TorchVision](https://pytorch.org/) libraries
- Nvidia GPU

#### 3. For Training
The model is trained using following command
```bash
python train.py batch_size=8 learning_rate=1e-5 num_train_epochs=50 dataset_dir=landscape_images/
```
Parameters and their default values:

>```batch_size```: **```8```** &nbsp; - &nbsp; batch size [can use large value depending upon resource availablility] <br/>
>```learning_rate```: **```1e-5```** &nbsp; - &nbsp; learning rate <br/>
>```num_train_epochs```: **```50```** &nbsp; - &nbsp; the number of training epochs <br/>
>```dataset_dir```: **```landscape_images/```** &nbsp; - &nbsp; path to the folder with **dataset for grayscale to rgb** <br/>
</br>

#### 4. For Inference
The model can be tested using following command
```bash
python test.py dataset_dir=landscape_images/ save_dir=test_result/ model_path=model/resnet_lab_full_epoch_14.pth
```
Parameters and their default values:

>```dataset_dir```: **```landscape_images/```** &nbsp; - &nbsp; path to the folder with **test dataset for grayscale to rgb** <br/>
>```save_dir```: **```test_result/```** &nbsp; - &nbsp; path to the folder with **where you want to save result** <br/>
>```model_path```: **```model/resnet_lab_full_epoch_14.pth```** &nbsp; - &nbsp; path to the  **model weight** <br/>
</br>

#### 5. Loss function Experimented with 
- L1 loss   
- L2 loss
- Smooth L1 Loss
- Multi-Scale Structural Simlarity Loss
After trying these loss following combination gave the best result for me 0.90 L1 + 0.1 SSIM

Given more time would like to further experiment with weighting L1 loss inversely to the frequency with which their colors appears and try to experiment with Perceptual Loss.

#### 6. Metric Used  
- L1 distance between generated image and actual image in ab color space 
- Deviation in average saturation color between generated image and actual image (to check average mood of image) in ab color space.


