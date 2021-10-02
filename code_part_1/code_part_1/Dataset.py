import argparse
import getpass
import imageio
import json
import os
import random
import torch
from Utils import get_clamped_psnr
import Utils as util
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import glob
from torch import nn
from math import sqrt
from torch.utils.data import Dataset, DataLoader
import torch
import tqdm # import tqdm
from collections import OrderedDict


class PolyU_DS(Dataset):

    def __init__(self, samples_dir_path):
     """
     Args:
            samples_dir_path (string): Path the dir with all image pairs (noisy, clean) ordered in folders.
        """
     self.samples_dir_path = samples_dir_path
     self.sample_numbers = [i for i in range(1, 101)]

    def __len__(self):
        return len(self.sample_numbers)

    def __getitem__(self, idx):
      sample_number = self.sample_numbers[idx]
      sample_path = self.samples_dir_path + "/" + str(sample_number) + "/*"
      samples_names = glob.glob(sample_path)
      real_img_path = samples_names[0] if "real" in samples_names[0] else samples_names[1]
      mean_img_path = samples_names[0] if "mean" in samples_names[0] else samples_names[1]
      real_img_tensor = transforms.ToTensor()(imageio.imread(real_img_path))
      mean_img_tensor = transforms.ToTensor()(imageio.imread(mean_img_path))

      return mean_img_tensor, real_img_tensor