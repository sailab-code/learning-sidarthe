import numpy as np
import random


def perturb(params, mu=0.0, sigma=1.0):
    for k, weights in params.items():
        params[k] = [w + random.gauss(mu, sigma) for w in weights]

    return params
