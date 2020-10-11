import os
from torch.utils.tensorboard import SummaryWriter
import torch

from experiments.sidarthe_extended_experiment import ExtendedSidartheExperiment
from learning_models.tied_sidarthe_extended import TiedSidartheExtended

from params.params import SidartheParamGenerator

if __name__ == "__main__":
    region = "Italy"
    n_epochs = 10000
    t_step = 1.0
    # train_size = 183
    experiment_cls = ExtendedSidartheExperiment  # switch class to change experiment: e.g. SidartheExperiment
    exp_path = os.path.join(os.getcwd(), "IT_68", "sidarthe_extended", region, "der_10000000.0_mTrue_dt_3.0_42c0f88e-dda4-46f2-9fc8-eea1198ee794")  # todo set me properly
    settings_json = experiment_cls.get_configs_from_json(os.path.join(exp_path, "settings.json"))
    loss_weights = settings_json["target_weights"]
    momentum = settings_json["momentum"]
    der_1st_reg = settings_json["der_1st_reg"]
    bound_reg = settings_json["bound_reg"]
    bound_loss_type = settings_json["bound_loss_type"]
    model_cls = TiedSidartheExtended  # todo load from settings if possible
    val_len = settings_json["val_len"]

    final_json = experiment_cls.get_configs_from_json(os.path.join(exp_path, "final.json"))
    initial_params = final_json["params"]

    runs_dir = "runs/sliding_train_exps"

    global_summary = SummaryWriter(f"{os.path.join(os.getcwd(), runs_dir)}")

    step = 15
    initial_train_size, final_train_size = settings_json["train_size"]+step, 203

    for train_size in range(initial_train_size, final_train_size, step):
        print(train_size)
        print(val_len)

        param_gen = SidartheParamGenerator()
        param_gen.init_from_base_params(initial_params)
        param_gen.extend(train_size)

        experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step, runs_directory=runs_dir, uuid_prefix=f"train_{train_size}_")
        model, uuid, res = experiment.run_exp(
            initial_params=param_gen.params,
            dataset_params={"train_size": train_size, "val_len": val_len},
            train_params={"momentum": momentum},
            model_params={"der_1st_reg": der_1st_reg, "bound_reg": bound_reg, "bound_loss_type": bound_loss_type, "model_cls": model_cls},
            loss_weights=loss_weights

        )

        # convert torch tensors into lists
        initial_params = {}
        for key in model.params.keys():
            initial_params[key] = model.params[key].tolist()
            # initial_params[key] = initial_params[key] + [initial_params[key][-1]] * (train_size - len(initial_params[key]))  # adds missing days


        # with torch.no_grad():
        #     # Plotting all together
        #     risks, hat_t, target_t, dataset_slice = experiment.compute_final_losses(x_target=experiment.dataset.inputs, targets=experiment.dataset.targets)
        #     experiment.plot_final_inferences(hat_t, target_t, dataset_slice, global_summary, collapse=True, prefix="all")

