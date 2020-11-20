import os
from datetime import datetime
import json

from pytorch_lightning import Callback


class JsonLoggingCallback(Callback):
    def __init__(self):
        """
        Callback to create a json description of the experimental setup.
        """
        pass

    def on_fit_start(self, trainer, pl_module):
        # creates the json description file with all trainer settings
        description = self._get_description(
            trainer.region, trainer.initial_params, trainer.learning_rates, trainer.model_params["loss_fn"],
            trainer.dataset_params["train_size"], trainer.dataset_params["val_size"],
            trainer.model_params["reg_fn"], trainer.time_step,
            trainer.train_params["momentum_settings"]["active"],
            trainer.train_params["momentum_settings"]["m"], trainer.train_params["momentum_settings"]["a"],
            trainer.model_params["integrator"],
            trainer.model_params["model_cls"]
        )

        json_description = json.dumps(description, indent=4)

        json_file = "settings.json"
        with open(os.path.join(trainer.exp_path, json_file), "a") as f:
            f.write(json_description)

    @staticmethod
    def _get_description(area, initial_params, learning_rates, loss_fn, train_size, val_size, reg_fn,
                         t_inc, momentum, m, a, integrator,
                         model_cls
                         ):
        return {
            "started": datetime.now().strftime('%d/%B/%Y %H:%M:%S'),
            "model_cls": str(model_cls),
            "region": area,
            "learning_rates": learning_rates,
            "loss_fn": str(loss_fn),
            "train_size": train_size,
            "val_size": val_size,
            "reg_fn": str(reg_fn),
            "integrator": integrator.__name__,
            "t_inc": t_inc,
            "momentum_settings": {
                "active": momentum,
                "m": m if momentum else None,
                "a": a if momentum else None
            },
            "initial_values": initial_params
        }
