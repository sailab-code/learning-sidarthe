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


exps_path = os.path.join(os.getcwd(), "fr_dec31")
if not os.path.exists(exps_path):
    os.mkdir(exps_path)

regions = ["FR"]

train_size, val_size = 188, 7
time_step = 1.0

data_path = os.path.join(os.getcwd(), "data", "dati-fr", "fr_data_processed.csv")
st_sidarthe_dataset = SpatioTemporalSidartheDataset(regions, data_path, train_size, val_size, "stato")
st_sidarthe_dataset.setup()

initial_params = {
    "alpha": [0.165] * train_size,
    "beta": [0.005] * train_size,
    "gamma": [0.10] * train_size,
    "delta": [0.005] * train_size,
    "epsilon": [0.1] * train_size,
    "theta": [0.18] * train_size,
    "zeta": [0.0034] * train_size,
    "eta": [0.0034] * train_size,
    "mu": [0.008] * train_size,
    "nu": [0.019] * train_size,
    "tau": [0.03] * train_size,
    "lambda": [0.07] * train_size,
    "kappa": [0.018] * train_size,
    "xi": [0.018] * train_size,
    "rho": [0.018] * train_size,
    "sigma": [0.02] * train_size,
    "phi": [0.02] * train_size,
    "chi": [0.02] * train_size
 }


ppls = [populations[area] for area in st_sidarthe_dataset.region]

model_params={
    "params": initial_params,
    "tied_parameters": {"delta": "beta",
                        "zeta": "eta",
                        },
    "population": ppls, # tensor of size S
    "initial_conditions": st_sidarthe_dataset.get_initial_conditions(ppls), # S x 8
    "integrator": Heun,
    "n_areas": st_sidarthe_dataset.n_areas,
    "loss_fn": NRMSE({
        "d": 0.035,
        "r": 0.025,
        "t": 0.02,
        "h": 0.015,
        "e": 0.02,
    }),
    "reg_fn": compose_losses(
        [
            LteZero(1e5),
            FirstDerivative(5e6, time_step)
        ]
    ),
    "time_step": time_step,
    "momentum_settings": {
        "b": 0.07,
        "a": 0.0,
        "active": True
    }
}
sidarthe_model = SpatioTemporalSidarthe(**model_params)

ckpt_path = os.path.join(exps_path,'checkpoints/weights.ckpt')
checkpoint_callback = ModelCheckpoint(filepath=ckpt_path, save_top_k=1, verbose=True, monitor='val_loss_unweighted', mode='min')
tb_logger = TensorBoardLogger(exps_path, name="tb_logs")
exp = CompartmentalTrainer(
    dataset=st_sidarthe_dataset,
    model=sidarthe_model,
    uuid_prefix="all_regions", uuid="",
    max_steps=3000,
    log_every_n_steps = 50,
    max_epochs=3000,
    default_root_dir=exps_path,
    check_val_every_n_epoch=50,
    gradient_clip_val=30.0,
    logger=tb_logger,
    checkpoint_callback=checkpoint_callback,
    callbacks=[JsonLoggingCallback(), PrintCallback(), TensorboardLoggingCallback()],
    # gpus=1,
)

exp.run_exp()