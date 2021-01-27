from pytorch_lightning import Trainer
from datetime import datetime

class CompartmentalTrainer(Trainer):
    def __init__(self, dataset, model, uuid, uuid_prefix, **kwargs):
        super().__init__(**kwargs)

        self.uuid = uuid
        self.uuid_prefix = uuid_prefix

        self.hyper_params_dict = {}
        self.dataset = dataset
        self.model = model

        self.exp_path = self.default_root_dir  # fixme vorrei tutto dentro lightning version

    def get_description(self):
        return {
            "started": datetime.now().strftime('%d/%B/%Y %H:%M:%S'),
            "region": self.dataset.region,
            "learning_rates": self.model.learning_rates,
            "loss_fn": str(self.model.loss_fn),
            "train_size": self.dataset.train_size,
            "val_size": self.dataset.val_size,
            "reg_fn": str(self.model.regularization_fn),
            "integrator": str(self.model.integrator),
            "t_inc": self.model.time_step,
            "momentum_settings": {
                "active": self.model.momentum_settings["active"],
                "b": self.model.momentum_settings["b"] if self.model.momentum_settings["active"] else None,
                "a": self.model.momentum_settings["a"] if self.model.momentum_settings["active"] else None
            },
            "initial_values": {k: v.detach().flatten().tolist() for k, v in self.model.params.items()}
        }

    def run_exp(self):
        """
        :return: trained_model, uuid, results
        """

        print(f"Running experiment {self.uuid}")
        self.fit(self.model, self.dataset.train_dataloader(), self.dataset.val_dataloader())
        self.test(self.model, self.dataset.test_dataloader(), ckpt_path="best")
