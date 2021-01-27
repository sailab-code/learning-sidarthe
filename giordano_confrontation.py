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

if __name__ == '__main__':
    exps_path = os.path.join(os.getcwd(), "giordano_confrontation")
    if not os.path.exists(exps_path):
        os.mkdir(exps_path)

    regions = ["ITA"]

    train_size, val_size = 39, 7
    time_step = 1.0

    data_path = os.path.join(os.getcwd(), "data", "giordano.csv")
    st_sidarthe_dataset = SpatioTemporalSidartheDataset(regions, data_path, train_size, val_size, "stato")
    st_sidarthe_dataset.setup()

    initial_params = {
        "alpha": [[0.57]] * 4 + [[0.4218]] * 18 + [[0.36]] * 6 + [[0.21]] * 11,
        "beta": [[0.0114]] * 4 + [[0.0057]] * 18 + [[0.005]] * 17,
        "gamma": [[0.456]] * 4 + [[0.285]] * 18 + [[0.2]] * 6 + [[0.11]] * 11,
        "delta": [[0.0114]] * 4 + [[0.0057]] * 18 + [[0.005]] * 17,
        "epsilon": [[0.171]] * 12 + [[0.1425]] * 26 + [[0.2]],
        "theta": [[0.3705]] * 39,
        "zeta": [[0.1254]] * 22 + [[0.034]] * 16 + [[0.025]],
        "eta": [[0.1254]] * 22 + [[0.034]] * 16 + [[0.025]],
        "mu": [[0.0171]] * 22 + [[0.008]] * 17,
        "nu": [[0.0274]] * 22 + [[0.015]] * 17,
        "tau": [[0.01]] * 39,
        "lambda": [[0.0342]] * 22 + [[0.08]] * 17,
        "rho": [[0.0342]] * 22 + [[0.0171]] * 16 + [[0.02]],
        "kappa": [[0.0171]] * 22 + [[0.0171]] * 16 + [[0.02]],
        "xi": [[0.0171]] * 22 + [[0.0171]] * 16 + [[0.02]],
        "sigma": [[0.0171]] * 22 + [[0.0171]] * 16 + [[0.01]],
        "phi": [[0.]],
        "chi": [[0.]]
    }

    lrates = {
        param_name: 1e-5 if param_name not in ["phi", "chi"] else 0.
        for param_name in initial_params.keys()
    }

    # check if all params are correct length
    assert(all([len(param) == train_size or len(param) == 1 for param_name, param in initial_params.items()]))

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
            "d": 0.05,
            "r": 0.01,
            "t": 0.01,
            "h": 0.01,
            "e": 0.01,
        }),
        "reg_fn": compose_losses(
            [
                LogVicinity(1.0),
                FirstDerivative(1e8, time_step)
            ]
        ),
        "time_step": time_step,
        "momentum_settings": {
            "b": 0.1,
            "a": 0.0,
            "active": True
        }
    }

    sidarthe_model = SpatioTemporalSidarthe(**model_params)
    ckpt_path = os.path.join(exps_path, 'checkpoints/weights.ckpt')
    checkpoint_callback = ModelCheckpoint(filepath=ckpt_path, save_top_k=1, verbose=True, monitor='val_loss_unweighted',
                                          mode='min')
    tb_logger = TensorBoardLogger(exps_path, name="tb_logs")
    exp = CompartmentalTrainer(
        dataset=st_sidarthe_dataset,
        model=sidarthe_model,
        uuid_prefix="all_regions", uuid="",
        max_steps=5000,
        log_every_n_steps=50,
        max_epochs=5000,
        default_root_dir=exps_path,
        check_val_every_n_epoch=50,
        gradient_clip_val=30.0,
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[JsonLoggingCallback(), PrintCallback(), TensorboardLoggingCallback()],
        # gpus=1,
    )
    version_path = os.path.join(exp.exp_path, exp.logger.name, f"version_{exp.logger.version}")

    # test on giordano's parameters
    losses, _ = exp.test(exp.model, exp.dataset.test_dataloader())

    with open(os.path.join(version_path, "loss_giordano.json"), "w+") as f:
        json.dump(losses, f)

    exp.run_exp()

    losses, _ = exp.test(exp.model, exp.dataset.test_dataloader())

    with open(os.path.join(version_path, "loss_lcm.json"), "w+") as f:
        json.dump(losses, f)
