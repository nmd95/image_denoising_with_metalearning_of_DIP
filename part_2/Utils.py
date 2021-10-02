from functools import partial
import jax
from jax import random, grad, jit, vmap, flatten_util, nn
import optax
from jax.config import config
import jax.numpy as np
import haiku as hk
import pickle

def process_example(example, RES):
    image = np.float32(example["image"]) / 255
    return image[:,image.shape[1]//2-RES//2:image.shape[1]//2+RES//2,image.shape[2]//2-RES//2:image.shape[2]//2+RES//2]