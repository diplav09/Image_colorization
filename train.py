import torch
import os
from torch.utils.data import DataLoader
from torch.optim import Adam
import random
import math
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self):
        # pass
        # Define hparams here or load them from a config file
        self.batch_size = 64
        self.learning_rate = 1e-5
        self.num_train_epochs = 10
        self.data_dir = '/content/gdrive/My Drive/image_colorization/landscape_images'
        self.device = torch.device("cuda")
        self.model = Net().to(self.device)
    def train(self):
        # pass
        # dataloaders
        torch.backends.cudnn.deterministic = True

        self.files_name = os.listdir(self.data_dir)
        random.shuffle(self.files_name)
        self.files_name = self.files_name[0:100]
        len_total_data = len(self.files_name)
        len_train_data = int(0.8 * len_total_data)
        train_list = self.files_name[0:len_train_data]
        val_list = self.files_name[len_train_data:]
        train_dataset = ColorizeData(train_list)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1,
                              pin_memory=True, drop_last=True)
        val_dataset = ColorizeData(val_list)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=1,
                              pin_memory=True, drop_last=False)
        # Model
        # model = torch.nn.DataParallel(model)
        # Loss function to use
        # criterion = 
        L1_loss = torch.nn.L1Loss()
        MS_SSIM = MSSSIM()
        # MSE_loss = torch.nn.MSELoss()
        # You may also use a combination of more than one loss function 
        # or create your own.
        optimizer = Adam(params=self.model.parameters(), lr=self.learning_rate)
        # train loop
        for epoch in range(self.num_train_epochs):
            torch.cuda.empty_cache()
            running_loss = 0.0
            running_l1 = 0.0
            running_msssim = 0.0
            train_iter = iter(train_dataloader)
            for i in range(len(train_dataloader)):
                optimizer.zero_grad()
                x, y = next(train_iter)
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                colorized = self.model(x)
                loss_l1 = L1_loss(colorized, y)
                loss_ssim = MS_SSIM(colorized, y)
                # total_loss = loss_l1 * 0.85 + loss_ssim * 0.15
                total_loss = loss_l1 
                running_l1 += loss_l1
                running_msssim += loss_ssim
                running_loss += total_loss
                total_loss.backward()
                optimizer.step()
                if i==len(train_dataloader)-1:
                    op = y.cpu().numpy()
                    op = op[0,:,:,:].squeeze()
                    op = np.transpose(op, (1, 2, 0))
                    plt.figure()
                    plt.imshow(op) 
                    plt.show()
                    ip = colorized.cpu().detach().numpy()
                    ip = ip[0,:,:,:].squeeze()
                    ip = np.transpose(ip, (1, 2, 0))
                    plt.figure()
                    print(np.max(op))
                    print(np.max(ip))
                    # plt.imshow(ip, cmap='gray', vmin=0, vmax=1)
                    plt.imshow(ip) 
                    plt.show()
                #     break
            # break
            print("Average train loss = %.3f" % (running_loss.item()))
            print("Average train l1 loss = %.3f" % (running_l1.item()))
            print("Average train msssim loss = %.3f" % (running_msssim.item()))
            loss_l1_eval, loss_psnr_eval, loss_ssim_eval = self.validate(val_dataloader)
            print("Average validation l1 loss = %.3f" % (loss_l1_eval))
            print("Average validation msssim loss = %.3f" % (loss_ssim_eval))
            print("Average validation PSNR = %.3f" % (loss_psnr_eval))

        


    def validate(self,val_dataloader):
    #     pass
        # Validation loop begin
        # ------
        # Validation loop end
        # ------
        # Determine your evaluation metrics on the validation dataset.
        L1_loss = torch.nn.L1Loss()
        MS_SSIM = MSSSIM()
        MSE_loss = torch.nn.MSELoss()
        self.model.eval()
        loss_l1_eval = 0
        loss_psnr_eval = 0
        loss_ssim_eval = 0
        TEST_SIZE = len(val_dataloader)
        with torch.no_grad():
            val_iter = iter(val_dataloader)
            for j in range(len(val_dataloader)):
                x, y = next(val_iter)
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                colorized = self.model(x)
                loss_mse_temp = MSE_loss(colorized, y).item()

                loss_psnr_eval += 20 * math.log10(1.0 / math.sqrt(loss_mse_temp))
                loss_ssim_eval += MS_SSIM(colorized, y)
                loss_l1_eval += L1_loss(colorized, y)

        loss_l1_eval = loss_l1_eval / TEST_SIZE      # to check color reproduction
        loss_psnr_eval = loss_psnr_eval / TEST_SIZE      # to check noise 
        loss_ssim_eval = loss_ssim_eval / TEST_SIZE      # to check reconstruction loss

        return loss_l1_eval, loss_psnr_eval, loss_ssim_eval  

