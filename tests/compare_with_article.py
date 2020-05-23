import pytest

import torch
from learning_models.sidarthe import Sidarthe
from torch_euler import Heun, euler
from matplotlib import pyplot as plt

params = {
    "alpha": [0.570] * 4 + [0.422] * 18 + [0.360] * 6 + [0.210] * 11,
    "beta": [0.011] * 4 + [0.0057] * 18 + [0.005] * 17,
    "gamma": [0.456] * 4 + [0.285] * 18 + [0.2] * 6 + [0.11] * 11,
    "delta": [0.011] * 4 + [0.0057] * 18 + [0.005] * 17,
    "epsilon": [0.171] * 12 + [0.143] * 26 + [0.2],
    "theta": [0.371] * 39,
    "xi": [0.125] * 22 + [0.034] * 16 + [0.025],
    "eta": [0.125] * 22 + [0.034] * 16 + [0.025],
    "mu": [0.017] * 22 + [0.008] * 17,
    "ni": [0.027] * 22 + [0.015] * 17,
    "tau": [0.01] * 39,
    "lambda": [0.034] * 22 + [0.08] * 17,
    "kappa": [0.017] * 22 + [0.017] * 16 + [0.02],
    "zeta": [0.017] * 22 + [0.017] * 16 + [0.02],
    "rho": [0.034] * 22 + [0.017] * 16 + [0.02],
    "sigma": [0.017] * 22 + [0.017] * 16 + [0.01]
}

i = 200./60e6
d = 20./60e6
a = 1./60e6
r = 2./60e6
t, h, e = 0., 0., 0.
s = 1. - (i + d + a + r + t + h + e)


initial_values = [
        s,
        i,
        d,
        a,
        r,
        t,
        e,
        h,
]

a_len = len(params["alpha"])
for key, value in params.items():
    assert len(value) == a_len, f"{key} has wrong size"


sidarthe = Sidarthe(params, 1, initial_values, euler, 1.,
                    d_weight=0.,
                    r_weight=0.,
                    t_weight=0.,
                    h_weight=0.,
                    e_weight=0.,
                    der_1st_reg=0.,
                    bound_reg=0.,
                    verbose=False,
                    loss_type="rmse"
                    )
with torch.no_grad():
    size = 45
    t_grid = torch.linspace(0, size, size)

    inference = sidarthe.inference(t_grid)
    #sol = inference["sol"]

    fig = plt.figure()

    for key in "idarthe":
            plt.plot(t_grid.numpy(), inference[key], label=key)

    plt.ylim((0, 0.001))
    plt.legend()
    fig.show()
