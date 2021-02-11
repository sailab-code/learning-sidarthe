from pytorch_lightning import Trainer
from datetime import datetime

from lcm.compartmental_model import CompartmentalModel
from lcm.datasets import ODEDataModule


class CompartmentalTrainer(Trainer):
    def __init__(self, dataset, model, uuid, uuid_prefix, **kwargs):
        super().__init__(**kwargs)

        self.uuid = uuid
        self.uuid_prefix = uuid_prefix

        self.hyper_params_dict = {}
        self.dataset: ODEDataModule = dataset
        self.model: CompartmentalModel = model

        self.exp_path = self.default_root_dir  # fixme vorrei tutto dentro lightning version

    def get_description(self):
        return {
            "trainer_class": self.__class__.__name__,
            "started": datetime.now().strftime('%d/%B/%Y %H:%M:%S'),
            "dataset": self.dataset.to_dict(),
            "model": self.model.get_description()
        }

    def run_exp(self):
        """
        :return: trained_model, uuid, results
        """

        print(f"Running experiment {self.uuid}")
        self.fit(self.model, self.dataset.train_dataloader(), self.dataset.val_dataloader())
        return self.test(None, self.dataset.test_dataloader(), ckpt_path="best", verbose=False)
