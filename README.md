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

