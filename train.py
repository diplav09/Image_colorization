import torch
import os
from torch.utils.data import DataLoader
from torch.optim import Adam
import random
import math
import matplotlib.pyplot as plt
from skimage import io, color
from msssim import MSSSIM
from colorize_data import ColorizeData
import math
import sys
from basic_model import Net

def process_command_args(arguments):

    # Specifying the default parameters

    batch_size = 50

    learning_rate = 1e-5

    num_train_epochs = None

    dataset_dir = './landscape_images/'

    for args in arguments:


        if args.startswith("batch_size"):
            batch_size = int(args.split("=")[1])

        if args.startswith("learning_rate"):
            learning_rate = float(args.split("=")[1])

        if args.startswith("num_train_epochs"):
            num_train_epochs = int(args.split("=")[1])

        if args.startswith("dataset_dir"):
            dataset_dir = args.split("=")[1]

    return batch_size, learning_rate, num_train_epochs, dataset_dir


# Processing command arguments

batch_size, learning_rate, num_train_epochs, dataset_dir = process_command_args(sys.argv)


class Trainer:
	def __init__(self):
		torch.cuda.empty_cache()
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.num_train_epochs = num_train_epochs
		self.data_dir = dataset_dir
		self.device = torch.device("cuda")
		self.model = Net().to(self.device)
	def train(self):

		torch.cuda.empty_cache()
		torch.backends.cudnn.deterministic = True

		#load the file list
		self.files_name = os.listdir(self.data_dir)
		#shuffle data for train and validation split
		random.shuffle(self.files_name)
		len_total_data = len(self.files_name)
		len_train_data = int(0.8 * len_total_data)
		train_list = self.files_name[0:len_train_data]
		val_list = self.files_name[len_train_data:]
		train_dataset = ColorizeData(train_list)
		train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
							  pin_memory=True, drop_last=True)
		val_dataset = ColorizeData(val_list)
		val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0,
							  pin_memory=True, drop_last=False)

		#loss function 
		L1_loss = torch.nn.L1Loss()
		# L1_loss = torch.nn.SmoothL1Loss(beta=5.0)
		MS_SSIM = MSSSIM()
		train_loss_l1 = []
		train_loss_msssim = []
		# train_loss = []
		val_loss_l1 = []
		val_loss_msssim = []
		val_loss_psnr = []
		min_val_loss = 100000.0

		#optimizer 
		
		optimizer = Adam(params=self.model.parameters(), lr=self.learning_rate)
		# train loop
		for epoch in range(self.num_train_epochs):

			running_loss = 0.0
			running_l1 = 0.0
			running_msssim = 0.0
			step =0 
			train_iter = iter(train_dataloader)
			for i in range(len(train_dataloader)):
				torch.cuda.empty_cache()
				optimizer.zero_grad()

				step+=1
				if(step%20==0):
					print(step)
				x, y = next(train_iter)
				x = x.to(self.device, non_blocking=True, dtype=torch.float)
				y = y.to(self.device, non_blocking=True, dtype=torch.float)
				colorized = self.model(x)

				loss_l1 = L1_loss(colorized, y)
				loss_ssim = MS_SSIM(colorized, y)
				#compute the training loss
				total_loss = loss_l1 * 0.95 + loss_ssim * 0.05
				
				running_l1 += loss_l1
				running_msssim += loss_ssim
				
				running_loss += total_loss
				total_loss.backward()
				optimizer.step()
				x = x.detach().cpu()
				y = y.detach().cpu()
		
			print("Average train loss = %.3f" % (running_loss.item()))
			print("Average train l1 loss = %.3f" % (running_l1.item()))
			print("Average train msssim loss = %.3f" % (running_msssim.item()))

			train_loss_l1.append(running_l1.item())
			train_loss_msssim.append(running_msssim.item())
			loss_l1_eval, loss_psnr_eval, loss_ssim_eval  = self.validate(val_dataloader)
			if(loss_l1_eval < min_val_loss):
				min_val_loss = loss_l1_eval
				torch.save(self.model.state_dict(), "./resnet_lab_full_epoch_" + str(epoch) + ".pth")
			val_loss_l1.append(loss_l1_eval)
			val_loss_msssim.append(loss_ssim_eval)
			val_loss_psnr.append(loss_psnr_eval)    
			print("Average validation l1 loss = %.3f" % (loss_l1_eval))
			print("Average validation msssim loss = %.3f" % (loss_ssim_eval))
			print("Average validation PSNR = %.3f" % (loss_psnr_eval))
		
		return train_loss_l1, train_loss_msssim, val_loss_l1, val_loss_msssim, val_loss_psnr

		


	def validate(self,val_dataloader):

		L1_loss = torch.nn.SmoothL1Loss(beta=5.0)
		MS_SSIM = MSSSIM()
		MSE_loss = torch.nn.MSELoss()
		self.model.eval()
		loss_l1_eval = 0
		loss_psnr_eval = 0
		loss_ssim_eval = 0
		cnt = 0
		TEST_SIZE = len(val_dataloader)
		with torch.no_grad():
			val_iter = iter(val_dataloader)
			for j in range(len(val_dataloader)):
				torch.cuda.empty_cache()
				x, y = next(val_iter)
				x = x.to(self.device, non_blocking=True, dtype=torch.float)
				y = y.to(self.device, non_blocking=True, dtype=torch.float)
				colorized = self.model(x)
				loss_mse_temp = MSE_loss(colorized, y).item()

				loss_psnr_eval += 20 * math.log10(1.0 / math.sqrt(loss_mse_temp))
				loss_ssim_eval += MS_SSIM(colorized, y)
				loss_l1_eval += L1_loss(colorized, y)*0.95 + loss_ssim_eval*0.05
				x = x.detach().cpu()
				y = y.detach().cpu()


		loss_l1_eval = loss_l1_eval / cnt      # to check color reproduction
		loss_psnr_eval = loss_psnr_eval / TEST_SIZE      # to check noise 
		loss_ssim_eval = loss_ssim_eval / TEST_SIZE      # to check reconstruction loss

		return loss_l1_eval, loss_psnr_eval, loss_ssim_eval

if __name__ == '__main__':
    training = Trainer()
    train_loss_l1, train_loss_msssim, val_loss_l1, val_loss_msssim, val_loss_psnr  = training.train()  

