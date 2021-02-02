import os
from datetime import datetime
import json

from pytorch_lightning import Callback

from lcm.compartmental_model import CompartmentalModel
from lcm.trainers import CompartmentalTrainer


class JsonLoggingCallback(Callback):
    def __init__(self):
        """
        Callback to create a json description of the experimental setup.
        """
        pass

    def on_train_start(self, trainer: CompartmentalTrainer, pl_module):
        # creates the json description file with all trainer settings
        description = trainer.get_description()

        json_description = json.dumps(description, indent=4)

        json_file = "settings.json"
        settings_path = os.path.join(trainer.exp_path, trainer.logger.name, f"version_{trainer.logger.version}",  json_file)
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        with open(settings_path, "a") as f:
            f.write(json_description)

    def on_train_end(self, trainer, pl_module: CompartmentalModel):
        final_params = pl_module.params

        np_params = {
            p_name: p_value.detach().cpu().flatten().tolist()
            for p_name, p_value in final_params.items()
        }

        json_file = "final_params.json"
        settings_path = os.path.join(trainer.exp_path, trainer.logger.name, f"version_{trainer.logger.version}", json_file)
        with open(settings_path, "a") as f:
            json.dump(np_params, f, indent=4)