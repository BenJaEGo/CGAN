import tensorflow as tf
import numpy as np


def sample_latent_variables_uniform(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def sample_latent_variables_normal(m, n):
    return np.random.normal(loc=0.0, scale=1.0, size=[m, n])


def fill_feed_dict(batch_data, latent_variables, model):
    x, y = batch_data
    feed_dict = dict()
    feed_dict[model.x_pl] = x
    feed_dict[model.z_pl] = latent_variables

    return feed_dict
