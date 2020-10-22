import os

from experiments.sidarthe_extended_experiment import ExtendedSidartheExperiment
from learning_models.tied_sidarthe_extended import TiedSidartheExtended

from params.params import SidartheParamGenerator

if __name__ == "__main__":
    region =  "FR"
    n_epochs = 0
    t_step = 1.0
    model_cls = TiedSidartheExtended
    val_size = 7

    experiment_cls = ExtendedSidartheExperiment  # switch class to change experiment: e.g. SidartheExperiment
    # exp_path = os.path.join(os.getcwd(), "runs", "IT_best", "sidarthe_extended", region, "_m0.1_no_missing_days_train_size188_der1000000.0_mTrue_dt_3.25_breg1.0_b5f4a743-30be-42d3-9eb8-513d2d59769f")
    # exp_path = os.path.join(os.getcwd(), "FR_exp", "best_FR")
    # exp_path = os.path.join(os.getcwd(), "runs", "IT_giordano_init", "sidarthe_extended", "Italy", "_m0.075_train_size188_der10000000.0_mTrue_dt_3.0_breg1.0_80956e9c-7241-4e03-8c42-63a421cbe733")
    # exp_path = os.path.join(os.getcwd(), "runs", "FR", "sidarthe_extended", "FR", "EPS_0_m0.1_ts163_der500000.0_dw_1.0_breg100000.0_2fe4124d-2226-4997-b524-b972eacea560")
    exp_path = os.path.join(os.getcwd(), "runs", "FR", "sidarthe_extended", "FR", "EPS_0_m0.1_ts163_der500000.0_dw_1.0_breg100000.0_fc7bb1ef-48d3-4fb5-a76b-9f3abae144f3")
    final_json = experiment_cls.get_configs_from_json(os.path.join(exp_path, "final.json"))
    settings_json = experiment_cls.get_configs_from_json(os.path.join(exp_path, "settings.json"))
    experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step, runs_directory="evals/FR_0.3", uuid_prefix=f"eval_res")

    param_gen = SidartheParamGenerator()
    param_gen.init_from_base_params(final_json["params"])
    param_gen.extend(166)
    initial_params = param_gen.params
    experiment.eval_exp(threshold=0.3,
                        initial_params=initial_params,
                        dataset_params={"train_size": 166, "val_len": val_size},
                        model_params={"model_cls": model_cls}
                        )
