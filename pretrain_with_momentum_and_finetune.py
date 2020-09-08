import os

from populations import populations
from torch_euler import Heun
from learning_models.sidarthe_extended import SidartheExtended
from experiments.sidarthe_experiment import SidartheExperiment
from experiments.sidarthe_extended_experiment import ExtendedSidartheExperiment

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # CHOOSE GPU HERE
if __name__ == "__main__":
    region = "Italy"
    t_step = 1.0

    # pretraining
    n_pretrain_epochs = 100
    pretrain_experiment_cls = ExtendedSidartheExperiment  # switch class to change experiment: e.g. SidartheExperiment
    pretrain_experiment = pretrain_experiment_cls(region, n_epochs=n_pretrain_epochs, time_step=t_step, runs_directory="runs")
    pretrained_model, uuid, res = pretrain_experiment.run_exp(model_params={"momentum": True})

    # finetuning
    print("Fine-tuning without momentum")

    n_finetune_epochs = 1000

    # convert torch tensor into list
    pretrained_params = {}
    for key in pretrained_model.params.keys():
        pretrained_params[key] = pretrained_model.params[key].tolist()
    print("Pretrained Params:")
    print(pretrained_params)

    fine_tune_experiment_cls = pretrain_experiment_cls
    fine_tune_experiment = fine_tune_experiment_cls(region, n_epochs=n_finetune_epochs, time_step=t_step, runs_directory="runs", uuid=uuid)
    model, uuid, res = fine_tune_experiment.run_exp(initial_params=pretrained_params, model_params={"momentum": False})

