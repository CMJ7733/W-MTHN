import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from val_data_functions import ValData
# from val_data_functions100h import ValData
from utils import validation
import os
import numpy as np
import random
from WEMT import WEMT_NET

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', default="cross_linear_best", type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
args = parser.parse_args()

val_batch_size = args.val_batch_size
exp_name = args.exp_name

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

# --- Set category-specific hyper-parameters  --- #
val_data_dir = '/root/autodl-tmp/data/test/'

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Validation data loader --- #

val_filename = 'filename_test.txt' ## This text file should contain all the names of the images and must be placed in ./data/test/ directory

val_data_loader = DataLoader(ValData(val_data_dir,val_filename), batch_size=val_batch_size, shuffle=False, num_workers=2)

# --- Define the network --- #

# net = Transweather()
net = Transweather_base().to(device)


net = nn.DataParallel(net, device_ids=device_ids)


# --- Load the network weight --- #
net.load_state_dict(torch.load('origin/best'))

# --- Use the evaluation model in testing --- #
net.eval()



if os.path.exists('./{}_results/'.format(exp_name))==False: 	
	os.mkdir('./{}_results/'.format(exp_name))	
print('--- Testing starts! ---')
start_time = time.time()
val_psnr, val_ssim = validation(net, val_data_loader, device, exp_name, save_tag=True)
end_time = time.time() - start_time
print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
print('validation time is {0:.4f}'.format(end_time))
