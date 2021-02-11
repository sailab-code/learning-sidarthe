import os
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


exps_path = os.path.join(os.getcwd(), "italy_jan21")
if not os.path.exists(exps_path):
    os.mkdir(exps_path)

regions = ["ITA"]

train_size, val_size = 305, 7
time_step = 1.0

data_path = os.path.join(os.getcwd(), "data", "COVID-19", "dati-andamento-nazionale", "dpc-covid19-ita-andamento-nazionale.csv")
st_sidarthe_dataset = SpatioTemporalSidartheDataset(regions, data_path, train_size, val_size, "stato")
st_sidarthe_dataset.setup()

initial_params = {
        "alpha": [[0.21]] * 4 + [[0.57]] * 18 + [[0.360]] * 6 + [[0.210]] * 10 + [[0.210]] * (train_size - 38),
        "beta": [[0.011]] * 4 + [[0.0057]] * 18 + [[0.005]] * (train_size - 22),
        "gamma": [[0.2]] * 4 + [[0.456]] * 18 + [[0.285]] * 6 + [[0.11]] * 10 + [[0.11]] * (train_size - 38),
        "delta": [[0.011]] * 4 + [[0.0057]] * 18 + [[0.005]] * (train_size - 22),
        "epsilon": [[0.171]] * 12 + [[0.143]] * 26 + [[0.2]] * (train_size - 38),
        "theta": [[0.371]] * train_size,
        "zeta": [[0.125]] * 22 + [[0.034]] * 16 + [[0.025]] * (train_size - 38),
        "eta": [[0.125]] * 22 + [[0.034]] * 16 + [[0.025]] * (train_size - 38),
        "mu": [[0.017]] * 22 + [[0.008]] * (train_size - 22),
        "nu": [[0.027]] * 22 + [[0.015]] * (train_size - 22),
        "tau": [[0.05]],
        "lambda": [[0.02]],
        "kappa": [[0.017]] * 22 + [[0.017]] * 16 + [[0.02]] * (train_size - 38),
        "xi": [[0.017]] * 22 + [[0.017]] * 16 + [[0.02]] * (train_size - 38),
        "rho": [[0.02]],
        "sigma": [[0.01]],
        "phi": [[0.02]] * train_size,
        "chi": [[0.02]] * train_size
    }

ppls = [populations[area] for area in st_sidarthe_dataset.region]

model_params={
    "params": initial_params,
    "tied_parameters": {"delta": "beta",
                        "lambda": "rho",
                        "kappa": "xi",
                        "zeta": "eta",
                        },
    "population": ppls, # tensor of size S
    "initial_conditions": st_sidarthe_dataset.get_initial_conditions(ppls), # S x 8
    "integrator": Heun,
    "n_areas": st_sidarthe_dataset.n_areas,
    "loss_fn": NRMSE({
        "d": 0.05,
        "r": 0.01,
        "t": 0.01,
        "h": 0.02,
        "e": 0.01,
    }, ignore_targets=None),  # ignore_targets=["d"] if you want to ignore target d
    "reg_fn": compose_losses(
        [
            LogVicinity(1.0),
            FirstDerivative(1e7, time_step)
        ]
    ),
    "time_step": time_step,
    "momentum_settings": {
        "b": 0.08,
        "a": 0.0,
        "active": True
    }
}
sidarthe_model = SpatioTemporalSidarthe(**model_params)

tb_logger = TensorBoardLogger(exps_path, name="tb_logs")
ckpt_path = os.path.join(exps_path, f'checkpoints/weights_{tb_logger.version}')
checkpoint_callback = ModelCheckpoint(filepath=ckpt_path, save_top_k=1, verbose=True, monitor='val_loss_unweighted', mode='min')

exp = CompartmentalTrainer(
    dataset=st_sidarthe_dataset,
    model=sidarthe_model,
    uuid_prefix="all_regions", uuid="",
    max_steps=5000,
    log_every_n_steps = 50,
    max_epochs=5000,
    default_root_dir=exps_path,
    check_val_every_n_epoch=50,
    gradient_clip_val=30.0,
    logger=tb_logger,
    checkpoint_callback=checkpoint_callback,
    callbacks=[JsonLoggingCallback(), PrintCallback(), TensorboardLoggingCallback()],
    # gpus=1,
)

exp.run_exp()