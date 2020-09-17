import os

from experiments.sidarthe_experiment import SidartheExperiment
from experiments.sidarthe_extended_experiment import ExtendedSidartheExperiment


if __name__ == "__main__":
    region = "Italy"
    n_epochs = 10000
    t_step = 1.0
    train_size = 183

    initial_params = {
        "alpha": [0.570] * 4 + [0.422] * 18 + [0.360] * 6 + [0.210] * 10 + [0.210] * (train_size - 38),
        "beta": [0.011] * 4 + [0.0057] * 18 + [0.005] * (train_size - 22),
        "gamma": [0.456] * 4 + [0.285] * 18 + [0.2] * 6 + [0.11] * 10 + [0.11] * (train_size - 38),
        "delta": [0.011] * 4 + [0.0057] * 18 + [0.005] * (train_size - 22),
        "epsilon": [0.171] * 12 + [0.143] * 26 + [0.2] * (train_size - 38),
        "theta": [0.371] * train_size,
        "zeta": [0.125] * 22 + [0.034] * 16 + [0.025] * (train_size - 38),
        "eta": [0.125] * 22 + [0.034] * 16 + [0.025] * (train_size - 38),
        "mu": [0.017] * 22 + [0.008] * (train_size - 22),
        "nu": [0.027] * 22 + [0.015] * (train_size - 22),
        "tau": [0.15],
        "lambda": [0.034] * 22 + [0.08] * (train_size - 22),
        # "lambda": [0.034],
        "kappa": [0.017] * 22 + [0.017] * 16 + [0.02] * (train_size - 38),
        # "kappa": [0.017],
        "xi": [0.017] * 22 + [0.017] * 16 + [0.02] * (train_size - 38),
        "rho": [0.034] * 22 + [0.017] * 16 + [0.02] * (train_size - 38),
        # "rho": [0.034],
        "sigma": [0.017] * 22 + [0.017] * 16 + [0.01] * (train_size - 38),
        "phi": [0.02] * train_size,
        "chi": [0.02] * train_size
    }

    loss_weights = {
        "d_weight": 5.,
        "r_weight": 1.,
        "t_weight": 1.,
        "h_weight": 1.,
        "e_weight": 1.
    }

    for k,v in loss_weights.items():
        loss_weights[k] = 0.002 * v

    experiment_cls = ExtendedSidartheExperiment  # switch class to change experiment: e.g. SidartheExperiment
    experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step, runs_directory="nature_init")
    experiment.run_exp(
        initial_params=initial_params,
        dataset_params={"train_size": train_size+4, "val_len": 10},
        train_params={"momentum": False},
        model_params={"der_1st_reg": 1e5, "bound_reg": 5e5, "bound_loss_type": "step"},
        loss_weights=loss_weights

    )  # params can be set, no params => default configuration

