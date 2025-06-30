import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_data_functions import TrainData
from val_data_functions import ValData
# from val_data_functions100h import ValData
from utils import to_psnr, print_log, validation, adjust_learning_rate
from torchvision.models import vgg16
from perceptual import LossNetwork
import os
import numpy as np
import random
import json 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from WMFormer import WMF_NET

plt.switch_backend('agg')

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=2e-4, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=48, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=8, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', default="rain200_cross_linear_net_pth", type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-num_epochs', help='number of epochs', default=1000, type=int)

args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs

os.makedirs(f"{exp_name}/plots", exist_ok=True)

training_history = {
    'train_loss': [],
    'smooth_loss': [],
    'perceptual_loss': [],
    'train_psnr': [],
    'val_psnr': [],
    'val_ssim': [],
    'epoch_times': []
}

# early stopping
class EarlyStopping:
    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_psnr):
        score = val_psnr

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

early_stopping = EarlyStopping(patience=20, delta=0.0)

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nlambda_loss: {}'.format(learning_rate, crop_size,
      train_batch_size, val_batch_size, lambda_loss))

train_data_dir = '/root/autodl-tmp/data/train/'
val_data_dir = '/root/autodl-tmp/data/test/'

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
net = WMF_NET()

# --- Build optimizer --- #
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)

# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)

scaler = torch.cuda.amp.GradScaler()

# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
for param in vgg_model.parameters():
    param.requires_grad = False

# --- Load the network weight --- #
if os.path.exists('{}/'.format(exp_name))==False:
    os.makedirs('{}/'.format(exp_name), exist_ok=True)
try:
    net.load_state_dict(torch.load('{}/best'.format(exp_name)))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')

loss_network = LossNetwork(vgg_model)
loss_network.eval()

# --- Load training data and validation/test data --- #
# labeled_name = 'rain100h_train.txt'
# val_filename1 = 'rain100h_test.txt'
labeled_name = 'filename_train.txt'
val_filename1 = 'filename_test.txt'

lbl_train_data_loader = DataLoader(TrainData(crop_size, train_data_dir,labeled_name), batch_size=train_batch_size, shuffle=True, num_workers=2)
val_data_loader1 = DataLoader(ValData(val_data_dir,val_filename1), batch_size=val_batch_size, shuffle=False, num_workers=0)

# --- Previous PSNR and SSIM in testing --- #
net.eval()
old_val_psnr1, old_val_ssim1 = validation(net, val_data_loader1, device, exp_name)
print('original_psnr: {0:.2f}, original_ssim: {1:.4f}'.format(old_val_psnr1, old_val_ssim1))
net.train()


def plot_training_history(history, exp_name):
    """绘制训练历史图表"""
    plt.figure(figsize=(15, 12))
    
    # 1. 
    plt.subplot(2, 2, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_psnr'], 'r*-', label='Validation PSNR')
    plt.title('Training Loss & Validation PSNR')
    plt.xlabel('Epochs')
    plt.legend()
    
    # 2. PSNR
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_psnr'], 'g^-', label='Training PSNR')
    plt.plot(epochs, history['val_psnr'], 'r*-', label='Validation PSNR')
    plt.title('Training and Validation PSNR')
    plt.xlabel('Epochs')
    plt.legend()
    
    # 3. LOSS
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['smooth_loss'], 'y--', label='Smooth Loss')
    plt.plot(epochs, history['perceptual_loss'], 'b--', label='Perceptual Loss')
    plt.plot(epochs, history['train_loss'], 'r-', label='Total Loss')
    plt.title('Detailed Loss Components')
    plt.xlabel('Epochs')
    plt.legend()
    
    # 4. SSIM
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['val_ssim'], 'm+-', label='Validation SSIM')
    plt.title('Validation SSIM')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{exp_name}/plots/training_history_epoch_{len(epochs)}.png")
    plt.close()

for epoch in range(epoch_start, num_epochs):
    psnr_list = []
    loss_list = []
    smooth_loss_list = []
    percep_loss_list = []
    
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch)
    
    for batch_id, train_data in enumerate(lbl_train_data_loader):
        input_image, gt, imgid = train_data
        input_image = input_image.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            pred_image = net(input_image)
            smooth_loss = F.smooth_l1_loss(pred_image, gt)
            perceptual_loss = loss_network(pred_image, gt)
            loss = smooth_loss + lambda_loss * perceptual_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_list.append(loss.item())
        smooth_loss_list.append(smooth_loss.item())
        percep_loss_list.append(perceptual_loss.item())
        psnr_list.extend(to_psnr(pred_image, gt))

        if batch_id % 100 == 0:
            print(f'Epoch: {epoch}, Iteration: {batch_id}, Loss: {loss.item():.4f}, Smooth Loss: {smooth_loss.item():.4f}, Percep Loss: {perceptual_loss.item():.4f}')

    train_psnr = sum(psnr_list) / len(psnr_list)
    avg_loss = sum(loss_list) / len(loss_list)
    avg_smooth_loss = sum(smooth_loss_list) / len(smooth_loss_list)
    avg_percep_loss = sum(percep_loss_list) / len(percep_loss_list)
    
    epoch_time = time.time() - start_time

    training_history['train_loss'].append(avg_loss)
    training_history['smooth_loss'].append(avg_smooth_loss)
    training_history['perceptual_loss'].append(avg_percep_loss)
    training_history['train_psnr'].append(train_psnr)
    training_history['epoch_times'].append(epoch_time)

    net.eval()
    val_psnr1, val_ssim1 = validation(net, val_data_loader1, device, exp_name)
    net.train()
    
    training_history['val_psnr'].append(val_psnr1)
    training_history['val_ssim'].append(val_ssim1)

    print_log(epoch+1, num_epochs, epoch_time, train_psnr, val_psnr1, val_ssim1, exp_name)

    torch.save(net.state_dict(), f'./{exp_name}/latest')
    if val_psnr1 >= old_val_psnr1:
        torch.save(net.state_dict(), f'./{exp_name}/best')
        print('Best model saved')
        old_val_psnr1 = val_psnr1

    if epoch % 5 == 0 or epoch == num_epochs - 1:
        with open(f"{exp_name}/training_history.json", "w") as f:
            json.dump(training_history, f, indent=4)
        plot_training_history(training_history, exp_name)

    early_stopping(val_psnr1)
    if early_stopping.early_stop:
        print("Early stopping triggered at epoch", epoch)
        break

plot_training_history(training_history, exp_name)
print("Training completed and history saved.")
