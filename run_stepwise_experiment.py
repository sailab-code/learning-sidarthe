import os
from collections import namedtuple

from learning_models.stepwise_tied_sidarthe_extended import StepwiseTiedSidartheExtended, Param
from experiments.sidarthe_extended_experiment import ExtendedSidartheExperiment


if __name__ == "__main__":
    region = "Italy"
    n_epochs = 10000
    t_step = 1.0
    train_size = 190

    # initial_params = {
    #     "alpha": [Param(0.570, 4), Param(0.422, 18), Param(0.360, 6), Param(0.210, 73), Param(0.210, train_size-101)],
    #     "beta": [Param(0.011, 4), Param(0.0057, 18), Param(0.005, train_size-38)],
    #     "gamma": [Param(0.456, 4), Param(0.285, 18), Param(0.2, 6), Param(0.11, 10), Param(0.11, train_size-38)],
    #     "delta": [Param(0.011, 4), Param(0.0057, 18), Param(0.005, train_size-38)],
    #     "epsilon": [Param(0.171, 12), Param(0.143, 26), Param(0.2, train_size - 38)],
    #     "theta": [Param(0.371, 38), Param(0.371, 63),  Param(0.371, train_size - 101)],
    #     "zeta": [Param(0.125, 22), Param(0.034, 16), Param(0.025, train_size - 38)],
    #     "eta": [Param(0.125, 22), Param(0.034, 16), Param(0.025, train_size - 38)],
    #     "mu": [Param(0.017, 22), Param(0.008, train_size - 22)],
    #     "nu": [Param(0.027, 22), Param(0.015, train_size - 22)],
    #     "tau": [Param(0.01, 4), Param(0.02, 18), Param(0.04, 6), Param(0.06, 10), Param(0.1, 30), Param(0.02, train_size - 68)],
    #     "lambda": [Param(0.034, 22), Param(0.08, train_size - 22)],
    #     # "lambda": [0.034],
    #     "kappa": [Param(0.017, 22), Param(0.017, 16), Param(0.02, train_size - 38)],
    #     # "kappa": [0.017],
    #     "xi": [Param(0.017, 22), Param(0.017, 16), Param(0.02, train_size - 38)],
    #     "rho": [Param(0.034, 22), Param(0.017, 16), Param(0.02, train_size - 38)],
    #     # "rho": [0.034],
    #     "sigma": [Param(0.017, 22), Param(0.017, 16), Param(0.01, train_size - 38)],
    #     "phi": [Param(0.02, 4), Param(0.02, 18), Param(0.02, 16), Param(0.02, train_size - 38)],
    #     "chi": [Param(0.02, 4), Param(0.02, 18), Param(0.02, 16), Param(0.02, train_size - 38)]
    # }

    # weekly
    wise_step = 7
    initial_params = {
        "alpha": [Param(0.570, 4), Param(0.422, 18), Param(0.360, 6), Param(0.210, 73)] + [Param(0.21, wise_step) for i in range((train_size-101)//wise_step)],
        "beta": [Param(0.011, 4), Param(0.0057, 18), Param(0.005, 16)] + [Param(0.005, wise_step) for i in range((train_size-38)//wise_step)],
        "gamma": [Param(0.456, 4), Param(0.285, 18), Param(0.2, 6), Param(0.11, 10)] + [Param(0.11, wise_step) for i in range((train_size-38)//wise_step)],
        "delta": [Param(0.011, 4), Param(0.0057, 18), Param(0.005, 16)] + [Param(0.005, wise_step) for i in range((train_size-38)//wise_step)],
        "epsilon": [Param(0.171, 12), Param(0.143, 26)] + [Param(0.2, wise_step) for i in range((train_size-38)//wise_step)],
        "theta": [Param(0.371, wise_step) for i in range((train_size)//wise_step)],
        # "zeta": [Param(0.125, 22), Param(0.034, 16)] + [Param(0.025, wise_step) for i in range((train_size-38)//wise_step)],
        # "eta":  [Param(0.125, 22), Param(0.034, 16)] + [Param(0.025, wise_step) for i in range((train_size-38)//wise_step)],
        # "mu": [Param(0.017, 22), Param(0.008, 16)] + [Param(0.008, wise_step) for i in range((train_size-38)//wise_step)],
        # "nu": [Param(0.027, 22), Param(0.015, 16)] + [Param(0.015, wise_step) for i in range((train_size-38)//wise_step)],
        # "tau": [Param(0.01, 4), Param(0.02, 18), Param(0.04, 6), Param(0.06, 10), Param(0.1, 30)] + [Param(0.025, wise_step) for i in range((train_size - 68)//wise_step)],
        # "lambda": [Param(0.034, 22), Param(0.08, 16)] + [Param(0.08, wise_step) for i in range((train_size-38)//wise_step)],
        #
        "zeta": [Param(0.125, wise_step) for _ in range(3)] + [Param(0.034, wise_step) for _ in range(3)] + [Param(0.025, wise_step) for i in range((train_size - 42) // wise_step)],
        "eta": [Param(0.125, wise_step) for _ in range(3)] + [Param(0.034, wise_step) for _ in range(3)] + [Param(0.025, wise_step) for i in range((train_size - 42) // wise_step)],
        "mu": [Param(0.017, wise_step) for _ in range(3)] + [Param(0.008, wise_step) for i in range((train_size - 21) // wise_step)],
        "nu": [Param(0.027, wise_step) for _ in range(3)] + [Param(0.015, wise_step) for i in range((train_size - 21) // wise_step)],
        "tau": [Param(0.01, 4), Param(0.02, 18), Param(0.04, 6), Param(0.06, 10), Param(0.1, 30)] + [Param(0.025, wise_step) for i in range((train_size - 68) // wise_step)],
        "lambda": [Param(0.034, wise_step) for _ in range(3)] + [Param(0.08, wise_step) for i in range((train_size - 21) // wise_step)],
        # "kappa": [Param(0.017, 22), Param(0.017, 16)] + [Param(0.02, wise_step) for i in range((train_size-38)//wise_step)],
        "kappa": [Param(0.02, wise_step) for i in range((train_size)//wise_step)],
        # "xi": [Param(0.017, 22), Param(0.017, 16)] + [Param(0.02, wise_step) for i in range((train_size-38)//wise_step)],
        "xi": [Param(0.02, wise_step) for i in range((train_size)//wise_step)],
        "rho": [Param(0.034, wise_step) for _ in range(3)] + [Param(0.08, wise_step) for i in range((train_size - 21) // wise_step)],
        "sigma": [Param(0.017, wise_step) for _ in range(6) for _ in range(3)] + [Param(0.01, wise_step) for i in range((train_size-42)//wise_step)],
        "phi": [Param(0.001, wise_step) for i in range((train_size)//wise_step)],
        "chi": [Param(0.001, wise_step) for i in range((train_size)//wise_step)]
    }

    loss_weights = {
        "d_weight": 3.5,
        "r_weight": 1.,
        "t_weight": 1.,
        "h_weight": 1.,
        "e_weight": 1.
    }

    for k,v in loss_weights.items():
        loss_weights[k] = 0.001 * v

    experiment_cls = ExtendedSidartheExperiment  # switch class to change experiment: e.g. SidartheExperiment
    experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step, runs_directory="stepwise_exps")
    experiment.run_exp(
        initial_params=initial_params,
        dataset_params={"train_size": train_size, "val_len": 22},
        train_params={"momentum": True, "a": 0.0},
        model_params={"model_cls": StepwiseTiedSidartheExtended, "der_1st_reg": 0, "bound_reg": 0, "bound_loss_type": "step"},
        loss_weights=loss_weights

    )  # params can be set, no params => default configuration

