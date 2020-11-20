import os
from pytorch_lightning.loggers import TensorBoardLogger

from lcm.trainers.sidarthe_trainer import SidartheTrainer
from lcm.callbacks.json_logging import JsonLoggingCallback
from lcm.callbacks.tensorboard_logging import TensorboardLoggingCallback
from lcm.callbacks.print_callback import PrintCallback

exps_path = os.path.join(os.getcwd(), "prova")
if not os.path.exists(exps_path):
    os.mkdir(exps_path)

tb_logger = TensorBoardLogger(exps_path, name="tb_logs")
exp = SidartheTrainer("ITA", 1., "prova", "",
                      max_steps=5,
                      default_root_dir=exps_path,
                      check_val_every_n_epoch=10,
                      gradient_clip_val=20.0,
                      logger=tb_logger,
                      callbacks=[JsonLoggingCallback(), PrintCallback(), TensorboardLoggingCallback()])
exp.run_exp(
    dataset_params={
        "train_size": 80
    }
)