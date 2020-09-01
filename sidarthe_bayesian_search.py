from typing import List
import os
import skopt

from populations import populations
from sidarthe_exp import exp
from torch_euler import Heun

if __name__ == '__main__':
    n_epochs = 4000
    region = "Italy"
    # params = {
    #     "alpha": [0.570] * 4 + [0.422] * 18 + [0.360] * 6 + [0.210] * 10 + [0.210],
    #     "beta": [0.011] * 4 + [0.0057] * 18 + [0.005] * 17,
    #     "gamma": [0.456] * 4 + [0.285] * 18 + [0.2] * 6 + [0.11] * 10 + [0.11],
    #     "delta": [0.011] * 4 + [0.0057] * 18 + [0.005] * 17,
    #     "epsilon": [0.171] * 12 + [0.143] * 26 + [0.2],
    #     "theta": [0.371],
    #     "zeta": [0.125] * 22 + [0.034] * 16 + [0.025],
    #     "eta": [0.125] * 22 + [0.034] * 16 + [0.025],
    #     "mu": [0.017] * 22 + [0.008] * 17,
    #     "nu": [0.027] * 22 + [0.015] * 17,
    #     "tau": [0.01],
    #     "lambda": [0.034] * 22 + [0.08] * 17,
    #     "kappa": [0.017] * 22 + [0.017] * 16 + [0.02],
    #     "xi": [0.017] * 22 + [0.017] * 16 + [0.02],
    #     "rho": [0.034] * 22 + [0.017] * 16 + [0.02],
    #     "sigma": [0.017] * 22 + [0.017] * 16 + [0.01]
    # }

    params = {
        "alpha": [0.570] * 39,
        "beta": [0.011] * 39,
        "gamma": [0.456] * 39,
        "delta": [0.011] * 39,
        "epsilon": [0.171] * 39,
        "theta": [0.371],
        "zeta": [0.125] * 39,
        "eta": [0.125] * 39,
        "mu": [0.017] * 39,
        "nu": [0.027] * 39,
        "tau": [0.01],
        "lambda": [0.034] * 39,
        "kappa": [0.017] * 39,
        "xi": [0.017] * 39,
        "rho": [0.034] * 39,
        "sigma": [0.017] * 39
    }

    learning_rates = {
        "alpha": 1e-5,
        "beta": 1e-5,
        "gamma": 1e-5,
        "delta": 1e-5,
        "epsilon": 1e-5,
        "theta": 1e-7,
        "xi": 1e-5,
        "eta": 1e-5,
        "mu": 1e-5,
        "nu": 1e-5,
        "tau": 1e-7,
        "lambda": 1e-5,
        "kappa": 1e-5,
        "zeta": 1e-5,
        "rho": 1e-5,
        "sigma": 1e-5
    }

    train_size = 46
    val_len = 20
    der_2nd_reg = 0.
    t_inc = 1.
    bound_reg = 1e4
    integrator = Heun
    loss_type = "rmse"

    SPACE = [
        skopt.space.Real(1e-1, 1e4, name="der_1st_reg", prior="log-uniform"),
        skopt.space.Real(0.05, 0.8, name="m", prior="uniform"),
        skopt.space.Real(0.01, 0.2, name="a", prior="uniform"),
        skopt.space.Real(1e-1, 1e1, name="d_w", prior="uniform"),
        skopt.space.Real(1e-1, 1e1, name="r_w", prior="uniform"),
        skopt.space.Real(1e-1, 1e1, name="t_w", prior="uniform"),
        skopt.space.Real(1e-1, 1e1, name="h_w", prior="uniform"),
        skopt.space.Categorical([True, False], name="momentum")
    ]

    results_dict = {}
    @skopt.utils.use_named_args(SPACE)
    def objective(**kwargs):

        loss_weights = {
            "d_weight": kwargs["d_w"],
            "r_weight": kwargs["r_w"],
            "t_weight": kwargs["t_w"],
            "h_weight": kwargs["h_w"]
        }

        momentum = kwargs["momentum"]
        m = kwargs["m"]
        a = kwargs["a"]
        der_1st_reg = kwargs["der_1st_reg"]

        exp_result = exp(
            region, populations[region], params,
            learning_rates, n_epochs, region, train_size, val_len,
            loss_weights, der_1st_reg, bound_reg, t_inc, integrator,
            momentum, m, a, loss_type
        )

        results_dict[exp_result[1]] = exp_result
        return exp_result[2].item()

    x0, y0 = None, None
    ckpt_folder = os.path.join(os.getcwd(), "bayesian_search_results")
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    ckpt_path = os.path.join(ckpt_folder, "results.pkl")
    if os.path.exists(ckpt_path):
        res_gp = skopt.load(ckpt_path)
        x0 = res_gp["x_iters"]
        y0 = res_gp["func_vals"]

    res_gp = skopt.gp_minimize(objective, SPACE, n_calls=10, n_jobs=-1, x0=x0, y0=y0)
    print(res_gp)
    skopt.dump(res_gp, ckpt_path)


