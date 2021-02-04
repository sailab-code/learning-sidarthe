import json
import os

import torch
from pytorch_lightning.loggers import TensorBoardLogger

from lcm.st_sidarthe import SpatioTemporalSidarthe
from lcm.datasets.st_sidarthe_dataset import SpatioTemporalSidartheDataset
from lcm.trainers import CompartmentalTrainer
from lcm.callbacks.json_logging import JsonLoggingCallback
from lcm.callbacks.tensorboard_logging import TensorboardLoggingCallback
from lcm.callbacks.print_callback import PrintCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from lcm.utils.populations import populations
from lcm.integrators.fixed_step import Heun
from lcm.losses import compose_losses
from lcm.losses.regularization_losses import LteZero, LogVicinity, FirstDerivative
from lcm.losses.target_losses import NRMSE

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # CHOOSE GPU HERE


def extract_losses(d):
    return {key: item
            for key, item in d.items()
            if "val_loss" in key}


if __name__ == '__main__':
    exps_path = os.path.join(os.getcwd(), "giordano_confrontation")
    if not os.path.exists(exps_path):
        os.mkdir(exps_path)

    regions = ["ITA"]

    train_size, val_size = 41, 5
    time_step = 1.0

    data_path = os.path.join(".", "data", "giordano.csv")
    st_sidarthe_dataset = SpatioTemporalSidartheDataset(regions, data_path, train_size, val_size, "stato")
    st_sidarthe_dataset.setup()

    initial_params = {
        "alpha": [[0.57]] * 4 + [[0.4218]] * 18 + [[0.36]] * 6 + [[0.21]] * 13,
        "beta": [[0.0114]] * 4 + [[0.0057]] * 18 + [[0.005]] * 19,
        "gamma": [[0.456]] * 4 + [[0.285]] * 18 + [[0.2]] * 6 + [[0.11]] * 13,
        "delta": [[0.0114]] * 4 + [[0.0057]] * 18 + [[0.005]] * 19,
        "epsilon": [[0.171]] * 12 + [[0.1425]] * 26 + [[0.2]] * 3,
        "theta": [[0.3705]] * 41,
        "zeta": [[0.1254]] * 22 + [[0.034]] * 16 + [[0.025]] * 3,
        "eta": [[0.1254]] * 22 + [[0.034]] * 16 + [[0.025]] * 3,
        "mu": [[0.0171]] * 22 + [[0.008]] * 19,
        "nu": [[0.0274]] * 22 + [[0.015]] * 19,
        "tau": [[0.01]] * 41,
        "lambda": [[0.0342]] * 22 + [[0.08]] * 19,
        "rho": [[0.0342]] * 22 + [[0.0171]] * 16 + [[0.02]] * 3,
        "kappa": [[0.0171]] * 22 + [[0.0171]] * 16 + [[0.02]] * 3,
        "xi": [[0.0171]] * 22 + [[0.0171]] * 16 + [[0.02]] * 3,
        "sigma": [[0.0171]] * 22 + [[0.0171]] * 16 + [[0.01]] * 3,
        "phi": [[0.01]],
        "chi": [[0.01]]
    }

    lrates = {
        param_name: 5e-5 if p_value[0][0] != 0. else 0.
        for param_name, p_value in initial_params.items()
    }

    # check if all params are correct length
    assert (all([len(param) == train_size or len(param) == 1 for param_name, param in initial_params.items()]))

    ppls = [populations[area] for area in st_sidarthe_dataset.region]

    # taken from giordano paper
    initial_conditions = []
    for ppl in ppls:
        i0 = 200
        d0 = 20
        a0 = 1
        r0 = 2
        t0 = 0
        h0 = 0
        e0 = 0
        s0 = ppl - i0 - d0 - a0 - r0 - t0 - h0 - e0
        initial_conditions.append(torch.tensor([[
            s0, i0, d0, a0, r0, t0, e0, h0
        ]]))

    initial_conditions = torch.cat(initial_conditions, dim=0).unsqueeze(0)
    x = st_sidarthe_dataset.get_initial_conditions(ppls)

    model_params = {
        "params": initial_params,
        "learning_rates": lrates,
        "EPS": 1e-12,
        "tied_parameters": {"delta": "beta",
                            "kappa": "xi",
                            "zeta": "eta",
                            },
        "population": ppls,  # tensor of size S
        "initial_conditions": initial_conditions,  # S x 8
        "integrator": Heun,
        "n_areas": st_sidarthe_dataset.n_areas,
        "loss_fn": NRMSE({
            "d": 0.6,
            "r": 0.7,
            "t": 0.8,
            "h": 0.4,
            "e": 0.8,
        }, ignore_targets=None),
        "reg_fn": compose_losses(
            [
                LogVicinity(1e-10),
                FirstDerivative(1e7, time_step)
            ]
        ),
        "time_step": time_step,
        "momentum_settings": {
            "b": 0.01,
            "a": 0.0,
            "active": True
        }
    }

    sidarthe_model = SpatioTemporalSidarthe(**model_params)
    tb_logger = TensorBoardLogger(exps_path, name="tb_logs")
    version_path = os.path.join(exps_path, tb_logger.name, f"version_{tb_logger.version}")
    ckpt_path = os.path.join(version_path, f'weights')
    checkpoint_callback = ModelCheckpoint(filepath=ckpt_path, save_top_k=1, verbose=True, monitor='val_loss_unweighted',
                                          mode='min')
    exp = CompartmentalTrainer(
        dataset=st_sidarthe_dataset,
        model=sidarthe_model,
        uuid_prefix="all_regions", uuid="",
        max_steps=1000,
        log_every_n_steps=25,
        max_epochs=1000,
        default_root_dir=exps_path,
        check_val_every_n_epoch=25,
        gradient_clip_val=50.0,
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[JsonLoggingCallback(), TensorboardLoggingCallback()],
        # gpus=1,
    )

    # test on giordano's parameters
    test_gior_out = exp.test(exp.model, exp.dataset.test_dataloader(), verbose=False)

    with open(os.path.join(version_path, "loss_giordano.json"), "w+") as f:
        json.dump(extract_losses(test_gior_out[0]), f, indent=4)

    test_lcm_out = exp.run_exp()

    tb_logger.experiment.close()

    with open(os.path.join(version_path, "loss_lcm.json"), "w+") as f:
        json.dump(extract_losses(test_lcm_out[0]), f, indent=4)
