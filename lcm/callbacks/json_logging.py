import os
from datetime import datetime
import json

from pytorch_lightning import Callback

from lcm.trainers import CompartmentalTrainer


class JsonLoggingCallback(Callback):
    def __init__(self):
        """
        Callback to create a json description of the experimental setup.
        """
        pass

    def on_fit_start(self, trainer: CompartmentalTrainer, pl_module):
        # creates the json description file with all trainer settings
        description = self._get_description(
            trainer.dataset.region, trainer.model.params, trainer.model.learning_rates, trainer.model.loss_fn,
            trainer.dataset.train_size, trainer.dataset.val_size,
            trainer.model.regularization_fn, trainer.model.time_step,
            trainer.model.momentum_settings["active"],
            trainer.model.momentum_settings["b"], trainer.model.momentum_settings["a"],
            trainer.model.integrator
        )

        json_description = json.dumps(description, indent=4)

        json_file = "settings.json"
        settings_path = os.path.join(trainer.exp_path, trainer.logger.name, f"version_{trainer.logger.version}",  json_file)
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        with open(settings_path, "a") as f:
            f.write(json_description)

    @staticmethod
    def _get_description(area, initial_params, learning_rates, loss_fn, train_size, val_size, reg_fn,
                         t_inc, momentum, m, a, integrator):

        return {
            "started": datetime.now().strftime('%d/%B/%Y %H:%M:%S'),
            "region": area,
            "learning_rates": learning_rates,
            "loss_fn": str(loss_fn),
            "train_size": train_size,
            "val_size": val_size,
            "reg_fn": str(reg_fn),
            "integrator": str(integrator),
            "t_inc": t_inc,
            "momentum_settings": {
                "active": momentum,
                "b": m if momentum else None,
                "a": a if momentum else None
            },
            "initial_values": {k: v.detach().tolist() for k,v in initial_params.items()}
        }
