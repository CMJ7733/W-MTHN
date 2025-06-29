import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
import os
import re

# --- Validation/test dataset --- #
class ValData(data.Dataset):
    def __init__(self, val_data_dir,val_filename):
        super().__init__()
        val_list = os.path.join(val_data_dir, val_filename)
        # val_list = val_data_dir + val_filename
        with open(val_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [
                re.sub(
                    r'/input/(\d+)_rain\.png$',  # 匹配input目录下的文件名
                    r'/gt/\g<1>_clean.png',       # 替换为gt目录且修改文件名
                    i.strip()
                ) 
                for i in contents
            ]

        self.input_names = input_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]

        input_img = Image.open(os.path.join(self.val_data_dir, input_name))
        gt_img = Image.open(os.path.join(self.val_data_dir, gt_name))
        # input_img = Image.open(self.val_data_dir + input_name)
        # gt_img = Image.open(self.val_data_dir + gt_name)
        # Resizing image in the multiple of 16"
        # wd_new,ht_new = input_img.size
        # if ht_new>wd_new and ht_new>1024:
        #     wd_new = int(np.ceil(wd_new*1024/ht_new))
        #     ht_new = 1024
        # elif ht_new<=wd_new and wd_new>1024:
        #     ht_new = int(np.ceil(ht_new*1024/wd_new))
        #     wd_new = 1024
        # wd_new = int(16*np.ceil(wd_new/16.0))
        # ht_new = int(16*np.ceil(ht_new/16.0))
        input_img = input_img.resize((720,480), Image.Resampling.LANCZOS)
        gt_img = gt_img.resize((720,480), Image.Resampling.LANCZOS)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_img)
        gt = transform_gt(gt_img)

        return input_im, gt, input_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
