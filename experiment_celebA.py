from functools import partial
import jax
from jax import random, grad, jit, vmap, flatten_util, nn
import optax
from jax.config import config
import jax.numpy as np
import haiku as hk
import pickle

# from livelossplot import PlotLosses
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds
from tqdm.notebook import tqdm as tqdm
import torch
import numpy as np_reg

from Siren import Siren_Model
from Utils import process_example
import argparse

RES = 178

def measure_psnr(ds, params, sigma, num_steps, random_seed=42):

  def outer_step(rng, image, coords, params, inner_steps, opt_inner):
      def loss_fn(params, rng_input):
          g = model.apply(params, coords)
          return mse_fn(g, image)
      
      image = np.reshape(image, (-1,3))
      coords = np.reshape(coords, (-1,2))
      opt_inner_state = opt_inner.init(params)
      loss = 0
      for _ in range(inner_steps):
          rng, rng_input = random.split(rng)
          loss, grad = jax.value_and_grad(loss_fn)(params, rng_input)

          updates, opt_inner_state = opt_inner.update(grad, opt_inner_state)
          params = optax.apply_updates(params, updates)
      return rng, params, loss
      
  log = {}

  model = hk.without_apply_rng(hk.transform(lambda x: Siren_Model()(x)))
  # params = model.init(random.PRNGKey(0), np.ones((1,2)))

  mse_fn = jit(lambda x, y: np.mean((x - y)**2))
  psnr_fn = jit(lambda mse: -10 * np.log10(mse))
  x1 = np.linspace(0, 1, RES+1)[:-1]
  coords = np.stack(np.meshgrid(x1,x1, indexing='ij'), axis=-1)[None,...]

  meta_opt_inner = optax.sgd(1e-2)
  nonmeta_opt_inner = optax.adam(1e-4)

  rng = random.PRNGKey(0)
  rng_test = random.PRNGKey(random_seed)
  params_non_meta = model.init(random.PRNGKey(0), np.ones((1,2)))
  for k, example in tqdm(enumerate(tfds.as_numpy(ds))):

    log_non_meta = []
    log_meta = []

    # print("\n image num:", i)

    test_img = process_example(example, RES)
    test_img_orig = test_img
    noise = np_reg.random.normal(scale=sigma, size=test_img.shape)
    noise = np.asarray(noise)
    test_img = test_img + noise
    test_img = np.clip(test_img, a_min=0.0, a_max=1.0)

    ref_psnr = psnr_fn(mse_fn(test_img_orig, test_img))
    # print("\n ref-psnr", ref_psnr)

    for i in range(num_steps):
        _, params_test, _ = outer_step(rng_test, test_img[None,...], coords, params, i, meta_opt_inner)
        render = model.apply(params_test, coords)[0]
        meta_psnr = psnr_fn(mse_fn(test_img_orig, np.clip(render,0,1)))
        log_meta.append(meta_psnr)
        # print("\n meta-psnr:", meta_psnr)

        _, params_test, _ = outer_step(rng_test, test_img[None,...], coords, params_non_meta, i, nonmeta_opt_inner)
        render = model.apply(params_test, coords)[0]
        non_meta_psnr = psnr_fn(mse_fn(test_img_orig, np.clip(render,0,1)))
        log_non_meta.append(non_meta_psnr)
        # print("\n non-meta-psnr:", non_meta_psnr)

    img_log = {"ref_psnr": ref_psnr, "log_meta": log_meta, "log_non_meta": log_non_meta}
    log[k] =  img_log
    # torch.save(log, res_save_dir_path + "/" + "img_" + str(i) + ".pkl")
  return log


def run_exp(ds, params, sigmas:list, num_steps, res_save_dir, random_seed=42):
  for sigma in sigmas:
    logz = measure_psnr(ds, params, sigma, num_steps, random_seed=random_seed)
    torch.save(logz, res_save_dir + "/" + "sigma_" + str(sigma) + ".pkl")

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='reproducing experiment_celebA')
  parser.add_argument('-mp','--meta_path', help='path to meta-initialization', required=True)
  parser.add_argument('-rp','--results_path', help='path of dir to which results will be saved', required=True)
  args = vars(parser.parse_args())

  meta_path = args['meta_path']
  results_path = args['results_path']

  gcs_base_dir = "gs://celeb_a_dataset/"
  builder = tfds.builder("celeb_a", data_dir=gcs_base_dir, version='2.0.0')
  builder.download_and_prepare()

  ds = builder.as_dataset(split='test', as_supervised=False, shuffle_files=False, batch_size=1)
  ds = ds.take(250)

  with open(meta_path, 'rb') as file:
    params = pickle.load(file)

  run_exp(ds=ds, params=params, sigmas=[0.1, 0.2, 0.4, 0.8, 1.0], num_steps=30, random_seed=42, res_save_dir=results_path)

