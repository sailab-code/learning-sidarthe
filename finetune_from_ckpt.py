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
from lcm.utils.checkpoint import load_from_json_checkpoint

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # CHOOSE GPU HERE


exps_path = os.path.join(os.getcwd(), "italy_dec31")
if not os.path.exists(exps_path):
    os.mkdir(exps_path)


if __name__ == '__main__':
    model, dataset = load_from_json_checkpoint(os.path.join(exps_path, "checkpoint.json"))
    exps_path = os.path.join(exps_path, "finetune")
    tb_logger = TensorBoardLogger(exps_path, name="tb_logs")
    ckpt_path = os.path.join(exps_path, f'checkpoints/weights_{tb_logger.version}')
    checkpoint_callback = ModelCheckpoint(filepath=ckpt_path, save_top_k=1, verbose=True, monitor='val_loss_unweighted',
                                          mode='min')

    exp = CompartmentalTrainer(
        dataset=dataset,
        model=model,
        uuid_prefix="all_regions", uuid="",
        max_steps=2000,
        log_every_n_steps=10,
        max_epochs=2000,
        default_root_dir=exps_path,
        check_val_every_n_epoch=10,
        gradient_clip_val=30.0,
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[JsonLoggingCallback(), PrintCallback(), TensorboardLoggingCallback()],
        # gpus=1,
    )

    exp.run_exp()