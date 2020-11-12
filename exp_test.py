from lcm.experiments.sidarthe_experiment import SidartheExperiment


exp = SidartheExperiment("ITA", 50, 1., "prova", "")
exp.run_exp(
    dataset_params={
        "train_size": 20
    }
)