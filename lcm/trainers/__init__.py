from pytorch_lightning import Trainer


class CompartmentalTrainer(Trainer):
    def __init__(self, dataset, model, uuid, uuid_prefix, **kwargs):
        super().__init__(**kwargs)

        self.uuid = uuid
        self.uuid_prefix = uuid_prefix

        self.hyper_params_dict = {}
        self.dataset = dataset
        self.model = model

        self.exp_path = self.default_root_dir  # fixme vorrei tutto dentro lightning version

    def run_exp(self):
        """
        :return: trained_model, uuid, results
        """

        print(f"Running experiment {self.uuid}")
        self.fit(self.model, self.dataset.train_dataloader(), self.dataset.val_dataloader())
        self.test(self.model, self.dataset.test_dataloader(), ckpt_path="best")
