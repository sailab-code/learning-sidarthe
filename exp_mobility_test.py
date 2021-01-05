import os
from pytorch_lightning.loggers import TensorBoardLogger

from lcm.trainers.mobility_trainer import MobilitySpatioTemporalSidartheTrainer
from lcm.callbacks.json_logging import JsonLoggingCallback
from lcm.callbacks.tensorboard_logging import TensorboardLoggingCallback
from lcm.callbacks.print_callback import PrintCallback
from pytorch_lightning.callbacks import ModelCheckpoint

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # CHOOSE GPU HERE


exps_path = os.path.join(os.getcwd(), "prova")
if not os.path.exists(exps_path):
    os.mkdir(exps_path)


checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(exps_path,'checkpoints/weights.ckpt'),
    save_top_k=1,
    verbose=True,
    monitor='val_loss_unweighted',
    mode='min'
)

tb_logger = TensorBoardLogger(exps_path, name="tb_logs")
exp = MobilitySpatioTemporalSidartheTrainer(
    "ITA", 1., "st_alpha", "",
    max_steps=4000,
    log_every_n_steps = 50,
    max_epochs=4000,
    default_root_dir=exps_path,
    check_val_every_n_epoch=50,
    gradient_clip_val=20.0,
    logger=tb_logger,
    checkpoint_callback=checkpoint_callback,
    callbacks=[JsonLoggingCallback(), PrintCallback(), TensorboardLoggingCallback()],
    # gpus=1,
)

regions = ["Lombardia", "Toscana"]

train_size = 150
initial_params = {
        "alpha": [[0.422]*len(regions)] * train_size,
        "beta": [[0.0057]] * train_size,
        "gamma": [[0.285]] * train_size,
        "delta": [[0.0057]] * train_size,
        "epsilon": [[0.143]] * train_size,
        "theta": [[0.371]] * train_size,
        "zeta": [[0.0034]] * train_size,
        "eta": [[0.0034]] * train_size,
        "mu": [[0.008]] * train_size,
        "nu": [[0.015]] * train_size,
        "tau": [[0.15]]* train_size,
        "lambda": [[0.08]] * train_size,
        "kappa": [[0.017]] * train_size,
        "xi": [[0.017]] * train_size,
        "rho": [[0.017]] * train_size,
        "sigma": [[0.017]] * train_size,
        "phi": [[0.02]] * train_size,
        "chi": [[0.02]] * train_size,
        "mobility0": [[2.]]
    }


exp.run_exp(
    initial_params= initial_params,
    dataset_params={
        "region": regions,
        "data_path": os.path.join(os.getcwd(), "data", "COVID-19", "dati-regioni", "dpc-covid19-ita-regioni.csv"),
        "train_size": train_size,
        "val_size": 20,
        "region_column": "denominazione_regione"
    },
    model_params={"tied_parameters": {"delta": "beta",
                                      "lambda": "rho",
                                      "kappa": "xi",
                                      "zeta": "eta"}
                  }
)