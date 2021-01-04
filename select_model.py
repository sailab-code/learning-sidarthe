import os

import numpy as np

from experiments.sidarthe_extended_experiment import ExtendedSidartheExperiment
from learning_models.tied_sidarthe_extended import TiedSidartheExtended

from params.params import SidartheParamGenerator

if __name__ == "__main__":
    # region = "Italy" # "FR"
    region = "FR"
    n_epochs = 0
    t_step = 1.0
    model_cls = TiedSidartheExtended

    train_size = 163  #188 # 185
    val_size = 7

    experiment_cls = ExtendedSidartheExperiment  # switch class to change experiment: e.g. SidartheExperiment

    exp_path = os.path.join(os.getcwd(), "runs", "FR", "sidarthe_extended", "FR")
    # exp_path = os.path.join(os.getcwd(), "runs", "IT_giordano_init", "sidarthe_extended", "Italy")
    exp_dirs = [dir for dir in os.listdir(exp_path)]
    exp_losses,exp_names = [],[]
    print(exp_dirs)
    for exp in exp_dirs:
        if os.path.isfile(os.path.join(exp_path, exp, "final.json")):
            final_json = experiment_cls.get_configs_from_json(os.path.join(exp_path, exp, "final.json"))
            settings_json = experiment_cls.get_configs_from_json(os.path.join(exp_path, exp, "settings.json"))

            param_gen = SidartheParamGenerator()
            param_gen.init_from_base_params(final_json["params"])
            param_gen.extend(166)
            initial_params = param_gen.params

            experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step, runs_directory="evals/FR_shortened", uuid_prefix=f"eval_res")
            exp_val_loss, max_r0 = experiment.eval_exp(
                threshold=0.3,
                initial_params=initial_params,
                dataset_params={"train_size": 166, "val_len": val_size},
                model_params={"model_cls": model_cls}
            )
            # exp_losses.append(exp_val_loss)
            if max_r0 < 10.:
                exp_names.append(exp)
                exp_losses.append(exp_val_loss)

    print(exp_losses)
    exp_losses = np.array(exp_losses)
    print(f"Best Experiment: {exp_names[np.argmin(exp_losses)]}")
    print(f"Val Loss: {exp_losses[np.argmin(exp_losses)]}")
