from functools import partial
import jax
from jax import random, grad, jit, vmap, flatten_util, nn
import optax
from jax.config import config
import jax.numpy as np
import haiku as hk
import pickle
import os
import torch
import numpy as np_reg


class SirenLayer(hk.Module):
    def __init__(self, in_f, out_f, w0=200, is_first=False, is_last=False):
        super().__init__()
        self.w0 = w0
        self.is_first = is_first
        self.is_last = is_last
        self.out_f = out_f
        self.b = 1 / in_f if self.is_first else np.sqrt(6 / in_f) / w0

    def __call__(self, x):
        x = hk.Linear(output_size=self.out_f, w_init=hk.initializers.RandomUniform(-self.b, self.b))(x)
        return x + .5 if self.is_last else self.w0 * x

class Siren_Model(hk.Module):
    def __init__(self):
        super().__init__()
        self.w0 = 200
        self.width = 256
        self.depth = 5        
        
    def __call__(self, coords):
        sh = coords.shape
        x = np.reshape(coords, [-1,2])
        x = SirenLayer(x.shape[-1], self.width, is_first=True, w0=self.w0)(x)
        x = np.sin(x)
        
        for _ in range(self.depth-2):
            x = SirenLayer(x.shape[-1], self.width, w0=self.w0)(x)
            x = np.sin(x)
            
        out = SirenLayer(x.shape[-1], 3, w0=self.w0, is_last=True)(x)
        out = np.reshape(out, list(sh[:-1]) + [3])
        return out