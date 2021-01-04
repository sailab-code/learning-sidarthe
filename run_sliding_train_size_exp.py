import os
from torch.utils.tensorboard import SummaryWriter
import torch

from experiments.sidarthe_extended_experiment import ExtendedSidartheExperiment
from learning_models.tied_sidarthe_extended import TiedSidartheExtended

from params.params import SidartheParamGenerator

if __name__ == "__main__":
    region =  "Italy" # "Italy"
    n_epochs = 10000
    t_step = 1.0

    experiment_cls = ExtendedSidartheExperiment  # switch class to change experiment: e.g. SidartheExperiment
    # exp_path = os.path.join(os.getcwd(), "IT_68", "sidarthe_extended", region, "der_10000000.0_mTrue_dt_3.0_42c0f88e-dda4-46f2-9fc8-eea1198ee794")
    # exp_path = os.path.join(os.getcwd(), "runs/FR_68", "sidarthe_extended", region, "545f8050-8d94-42bf-aae5-c1669a17e112") #fixme setting
    # settings_json = experiment_cls.get_configs_from_json(os.path.join(exp_path, "settings.json"))

    loss_weights = {
        "d_weight": 3.0,
        "r_weight": 1.0,
        "t_weight": 1.0,
        "h_weight": 1.0,
        "e_weight": 1.0
    }

    for k, v in loss_weights.items():
        loss_weights[k] = 0.005 * v  # settings_json["target_weights"]

    # momentum = settings_json["momentum"]
    # m = settings_json["m"]
    der_1st_reg = 1e7  #settings_json["der_1st_reg"]
    bound_reg =  1.0  #settings_json["bound_reg"]
    bound_loss_type = "log" #settings_json["bound_loss_type"]
    model_cls = TiedSidartheExtended
    #settings_json["val_len"]

    train_size = 68
    # initial_params = {
    #     "alpha": [0.422] * train_size,
    #     "beta": [0.0057] * train_size,
    #     "gamma": [0.285] * train_size,
    #     "delta": [0.0057] * train_size,
    #     "epsilon": [0.143] * train_size,
    #     "theta": [0.371] * train_size,
    #     "zeta": [0.0034] * train_size,
    #     "eta": [0.0034] * train_size,
    #     "mu": [0.008] * train_size,
    #     "nu": [0.015] * train_size,
    #     "tau": [0.15] * train_size,
    #     "lambda": [0.08] * train_size,
    #     "kappa": [0.017] * train_size,
    #     "xi": [0.017] * train_size,
    #     "rho": [0.017] * train_size,
    #     "sigma": [0.017] * train_size,
    #     "phi": [0.02] * train_size,
    #     "chi": [0.02] * train_size
    # }

    initial_params = {
            "alpha": [0.210] * 4 + [0.570] * 18 + [0.360] * 6 + [0.210] * 10 + [0.210] * (train_size - 38),
            "beta": [0.011] * 4 + [0.0057] * 18 + [0.005] * (train_size - 22),
            "gamma": [0.2] * 4 + [0.456] * 18 + [0.285] * 6 + [0.11] * 10 + [0.11] * (train_size - 38),
            "delta": [0.011] * 4 + [0.0057] * 18 + [0.005] * (train_size - 22),
            "epsilon": [0.171] * 12 + [0.143] * 26 + [0.2] * (train_size - 38),
            "theta": [0.371] * train_size,
            "zeta": [0.125] * 22 + [0.034] * 16 + [0.025] * (train_size - 38),
            "eta": [0.125] * 22 + [0.034] * 16 + [0.025] * (train_size - 38),
            "mu": [0.017] * 22 + [0.008] * (train_size - 22),
            "nu": [0.027] * 22 + [0.015] * (train_size - 22),
            "tau": [0.05]*train_size,
            "lambda": [0.034] * 22 + [0.08] * (train_size - 22),
            "kappa": [0.017] * 22 + [0.017] * 16 + [0.02] * (train_size - 38),
            "xi": [0.017] * 22 + [0.017] * 16 + [0.02] * (train_size - 38),
            "rho": [0.034] * 22 + [0.017] * 16 + [0.02] * (train_size - 38),
            "sigma": [0.017] * 22 + [0.017] * 16 + [0.01] * (train_size - 38),
            "phi": [0.02] * train_size,
            "chi": [0.02] * train_size
        }

    # final_json = experiment_cls.get_configs_from_json(os.path.join(exp_path, "final.json"))
    # initial_params = final_json["params"]

    runs_dir = "runs/IT/sliding_train_exps"

    global_summary = SummaryWriter(f"{os.path.join(os.getcwd(), runs_dir)}")

    step = 15
    # initial_train_size, final_train_size = settings_json["train_size"]+step, 203
    initial_train_size, final_train_size = 68, 203

    for train_size in range(initial_train_size, final_train_size, step):
        print(train_size)

        param_gen = SidartheParamGenerator()
        param_gen.init_from_base_params(initial_params)
        param_gen.extend(train_size)

        experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step, runs_directory=runs_dir, uuid_prefix=f"train_{train_size}_")
        model, uuid, res = experiment.run_exp(
            initial_params=param_gen.params,
            dataset_params={"train_size": train_size, "val_len": 20},
            train_params={"momentum": True, "m": 0.125, "a": 0.0},
            model_params={"der_1st_reg": der_1st_reg, "bound_reg": bound_reg, "bound_loss_type": bound_loss_type, "model_cls": model_cls},
            loss_weights=loss_weights

        )

        # convert torch tensors into lists
        initial_params = {}
        for key in model.params.keys():
            initial_params[key] = model.params[key].tolist()

