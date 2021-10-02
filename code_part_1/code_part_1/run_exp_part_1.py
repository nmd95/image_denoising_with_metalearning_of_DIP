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
from Sirens import Siren, Trainer
from Dataset import PolyU_DS

dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')


def run_experiment(ds, pkls_save_dir_path:str, arch:list, num_steps:int):
  num_steps = num_steps

  for i in range(1, 101):
    print("i:", i)
    sample = ds[i]
    img, noisy_img = sample[0].float().to(device, dtype), sample[1].float().to(device, dtype)

    ref_psnr = get_clamped_psnr(noisy_img.cpu(), img.cpu())

    print("\n ref_psnr:", ref_psnr)
    
    func_rep = Siren(
        dim_in=2,
        dim_hidden=arch[0],
        dim_out=3,
        num_layers=arch[1],
        final_activation=torch.nn.Identity(),
        w0_initial=30.0,
        w0=30.0 
    ).to(device)

    bpp = util.bpp(model=func_rep, image=img)

    trainer = Trainer(func_rep, lr=2e-4, img_shape=img.shape)
    coordinates, features = util.to_coordinates_and_features(img)
    _, features_noisy = util.to_coordinates_and_features(noisy_img)
    coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)
    features_noisy = features_noisy.to(device, dtype)

    trainer.train(coordinates, features, features_noisy=features_noisy,num_iters=num_steps)

    torch.save([bpp, ref_psnr, trainer], pkls_save_dir_path + "_" + str(i + 1) + ".pt")


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='meta-init-training')
  parser.add_argument('-rp','--results_path', help='path of dir to which results will be saved', required=True)
  parser.add_argument('-ip','--images_path', help='path of dir in which images are saved', required=True)

  args = vars(parser.parse_args())

  rp = args['results_path']
  ip = args['images_path']

  polyu_ds = PolyU_DS(samples_dir_path=ip)
  run_experiment(ds=polyu_ds, arch=[256, 4], num_steps=2000, pkls_save_dir_path=rp)

  results_path = args['results_path']