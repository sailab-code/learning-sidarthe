from experiments.sidarthe_extended_experiment import ExtendedSidartheExperiment
from learning_models.tied_sidarthe_extended import TiedSidartheExtended

if __name__ == "__main__":
    region = "Italy"  # "Italy"
    n_epochs = 10000
    t_step = 1.0
    train_size = 185  # 185

    # initial_params = {
    #     "alpha": [0.570] * 4 + [0.422] * 18 + [0.360] * 6 + [0.210] * 10 + [0.210] * (train_size - 38),
    #     "beta": [0.011] * 4 + [0.0057] * 18 + [0.005] * (train_size - 22),
    #     "gamma": [0.456] * 4 + [0.285] * 18 + [0.2] * 6 + [0.11] * 10 + [0.11] * (train_size - 38),
    #     "delta": [0.011] * 4 + [0.0057] * 18 + [0.005] * (train_size - 22),
    #     "epsilon": [0.171] * 12 + [0.143] * 26 + [0.2] * (train_size - 38),
    #     "theta": [0.371] * train_size,
    #     "zeta": [0.125] * 22 + [0.034] * 16 + [0.025] * (train_size - 38),
    #     "eta": [0.125] * 22 + [0.034] * 16 + [0.025] * (train_size - 38),
    #     "mu": [0.017] * 22 + [0.008] * (train_size - 22),
    #     "nu": [0.027] * 22 + [0.015] * (train_size - 22),
    #     "tau": [0.15]*train_size,
    #     "lambda": [0.034] * 22 + [0.08] * (train_size - 22),
    #     "kappa": [0.017] * 22 + [0.017] * 16 + [0.02] * (train_size - 38),
    #     "xi": [0.017] * 22 + [0.017] * 16 + [0.02] * (train_size - 38),
    #     "rho": [0.034] * 22 + [0.017] * 16 + [0.02] * (train_size - 38),
    #     "sigma": [0.017] * 22 + [0.017] * 16 + [0.01] * (train_size - 38),
    #     "phi": [0.02] * train_size,
    #     "chi": [0.02] * train_size
    # }

    initial_params = {
        "alpha": [0.422] * train_size,
        "beta": [0.0057] * train_size,
        "gamma": [0.285] * train_size,
        "delta": [0.0057] * train_size,
        "epsilon": [0.143] * train_size,
        "theta": [0.371] * train_size,
        "zeta": [0.0034] * train_size,
        "eta": [0.0034] * train_size,
        "mu": [0.008] * train_size,
        "nu": [0.015] * train_size,
        "tau": [0.15] * train_size,
        "lambda": [0.08] * train_size,
        "kappa": [0.017] * train_size,
        "xi": [0.017] * train_size,
        "rho": [0.017] * train_size,
        "sigma": [0.017] * train_size,
        "phi": [0.02] * train_size,
        "chi": [0.02] * train_size
    }

    # France
    # initial_params = {
    #     "alpha": [0.165] * train_size,
    #     "beta": [0.005] * train_size,
    #     "gamma": [0.10] * train_size,
    #     "delta": [0.005] * train_size,
    #     "epsilon": [0.1] * train_size,
    #     "theta": [0.18] * train_size,
    #     "zeta": [0.0034] * train_size,
    #     "eta": [0.0034] * train_size,
    #     "mu": [0.008] * train_size,
    #     "nu": [0.019] * train_size,
    #     "tau": [0.03] * train_size,
    #     "lambda": [0.07] * train_size,
    #     "kappa": [0.018] * train_size,
    #     "xi": [0.018] * train_size,
    #     "rho": [0.018] * train_size,
    #     "sigma": [0.02] * train_size,
    #     "phi": [0.02] * train_size,
    #     "chi": [0.02] * train_size
    # }

    # loss_weights = {
    #     "d_weight": 5.,
    #     "r_weight": 1.,
    #     "t_weight": 1.,
    #     "h_weight": 1.,
    #     "e_weight": 1.
    # }
    #
    # for k,v in loss_weights.items():
    #     loss_weights[k] = 0.02 * v

    loss_weights = {
        "d_weight": 3.0,
        "r_weight": 1.0,
        "t_weight": 1.0,
        "h_weight": 1.0,
        "e_weight": 1.0
    }

    for k, v in loss_weights.items():
        loss_weights[k] = 0.02 * v

    experiment_cls = ExtendedSidartheExperiment  # switch class to change experiment: e.g. SidartheExperiment
    experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step, runs_directory="runs/IT_constant")
    experiment.run_exp(
        initial_params=initial_params,
        dataset_params={"train_size": train_size, "val_len": 20},
        train_params={"momentum": True, "m": 0.1, "a": 0.0},
        # train_params={"momentum": True, "m": 0.05, "a": 0.0},
        # model_params={"der_1st_reg": 1e7, "bound_reg": 1.0, "bound_loss_type": "log", "model_cls": TiedSidartheExtended},
        model_params={"der_1st_reg": 1e7, "bound_reg": 1000000.0, "bound_loss_type": "step", "model_cls": TiedSidartheExtended},
        loss_weights=loss_weights

    )  # params can be set, no params => default configuration

