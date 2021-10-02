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
import argparse

from Utils import process_example
from Siren import Siren_Model


RES = 178

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

def update_model(rng, params, opt_state, image, coords, inner_steps, opt_inner):
  def loss_model(params, rng):
      rng = random.split(rng, batch_size)
      rng, new_params, loss = outer_step_batch(rng, image, coords, params, inner_steps, opt_inner)
      g = vmap(model.apply, in_axes=[0,None])(new_params, coords)
      return mse_fn(g[:,0,...], image), rng[0]
  (loss, rng), model_grad = jax.value_and_grad(loss_model, argnums=0, has_aux=True)(params, rng)

  updates, opt_state = opt_outer.update(model_grad, opt_state)
  params = optax.apply_updates(params, updates)
  return rng, params, opt_state, loss

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='meta-init-training')
  parser.add_argument('-rp','--results_path', help='path of dir to which results will be saved', required=True)
  args = vars(parser.parse_args())

  results_path = args['results_path']



  num_val_exs = 5
  batch_size = 3
  RES = 178

  outer_step_batch = vmap(outer_step, in_axes=[0, 0, None, None, None, None])
  
  gcs_base_dir = "gs://celeb_a_dataset/"
  builder = tfds.builder("celeb_a", data_dir=gcs_base_dir, version='2.0.0')
  builder.download_and_prepare()

  ds = builder.as_dataset(split='train', as_supervised=False, shuffle_files=True, batch_size=3)
  ds = ds.take(-1)

  ds_val = builder.as_dataset(split='validation', as_supervised=False, shuffle_files=False, batch_size=1)
  ds_val = ds_val.take(num_val_exs)

  meta_method = "MAML" 

  max_iters = 150000

  outer_lr = 1e-5 
  inner_lr = .01 
  inner_steps = 2 

  x1 = np.linspace(0, 1, RES+1)[:-1]
  coords = np.stack(np.meshgrid(x1,x1, indexing='ij'), axis=-1)[None,...]

  model = hk.without_apply_rng(hk.transform(lambda x: Siren_Model()(x)))
  params = model.init(random.PRNGKey(0), np.ones((1,2)))

  opt_outer = optax.adam(outer_lr)
  opt_inner = optax.sgd(inner_lr)

  opt_state = opt_outer.init(params)

  mse_fn = jit(lambda x, y: np.mean((x - y)**2))
  psnr_fn = jit(lambda mse: -10 * np.log10(mse))

  train_psnrs = []
  train_psnr = []
  val_psnrs = []
  steps = []
  step = 0
  rng = random.PRNGKey(0)
  rng_test = random.PRNGKey(42)
  while step < max_iters:
      for example in tqdm(tfds.as_numpy(ds), leave=False, desc='iter'):
          if step > max_iters:
              break

          image = process_example(example, RES)

          rng, params, opt_state, loss = update_model(rng, params, opt_state, image, coords, inner_steps, opt_inner)
          train_psnr.append(psnr_fn(loss))

          if step % 200 == 0 and step != 0:
              train_psnrs.append(np.mean(np.array(train_psnr)))
              train_psnr = []
              val_psnr = []
              for val_example in tfds.as_numpy(ds_val):
                  val_img = process_example(val_example, RES)
                  _, params_test, loss = outer_step(rng_test, val_img, coords, params, inner_steps, opt_inner)
                  img = model.apply(params_test, coords)[0]
                  val_psnr.append(psnr_fn(mse_fn(img, val_img)))
              val_psnrs.append(np.mean(np.array(val_psnr)))

              steps.append(step)
          step += 1
  torch.save(params, rp + "/META_WEIGHTS.pkl")

