import os
from pytorch_lightning import Trainer

from lcm.compartmental_model import CompartmentalModel


class CompartmentalTrainer(Trainer):
    def __init__(self, region, time_step, uuid, uuid_prefix, **kwargs):
        super().__init__(**kwargs)
        self.region = region
        self.time_step = time_step

        self.uuid = uuid
        self.uuid_prefix = uuid_prefix

        self.hyper_params_dict = {}
        self.dataset = None

        self.initial_params = None
        self.dataset_params = None
        self.model_params = None
        self.learning_rates = None
        self.train_params = None
        self.references = None
        self.model = None

        self.exp_path = self.default_root_dir  # fixme vorrei tutto dentro lightning version

    def run_exp(self, **kwargs):
        """

        :param kwargs: Optional arguments. Expected (optional) attributes
            {'initial_params': dict,
            'model_params': dict,
            'dataset_params': dict,
            'train_params': dict,
            'learning_rates': dict,
            'loss_weights': dict,
            }

        :return: trained_model, uuid, results
        """
        # creates initial params
        initial_params = self.make_initial_params(**kwargs)
        self.set_initial_params(initial_params)

        # creates dataset params
        dataset_params = self.make_dataset_params(**kwargs)
        self.set_dataset_params(dataset_params)

        # gets the data for the pretrained_model
        dataset_cls = self.dataset_params["dataset_cls"]
        self.dataset = dataset_cls(
            region=self.dataset_params['region'],
            data_path=self.dataset_params['data_path'],
            train_size=self.dataset_params['train_size'],
            val_size=self.dataset_params['val_size'],
            region_column=self.dataset_params["region_column"]
        )
        self.dataset.setup()

        # create references
        references = self.make_references(**kwargs)
        self.set_references(references)

        train_params = self.make_train_params(**kwargs)
        self.set_train_params(train_params)

        learning_rates = self.make_learning_rates(**kwargs)
        self.set_learning_rates(learning_rates)

        model_params = self.make_model_params(**kwargs)
        self.set_model_params(model_params)

        model = self.make_model(**kwargs)
        self.set_model(model)

        print(f"Running experiment {self.uuid}")
        # trainer = Trainer(check_val_every_n_epoch=50)
        self.fit(model, self.dataset.train_dataloader(), self.dataset.val_dataloader())
        self.test(model, self.dataset.test_dataloader())


    @staticmethod
    def fill_missing_params(params, default_params):
        """
        Fill attributes of params that are missing with
        the default values.
        :param params: dict of parameters
        :param default_params: dict of default parameters
        :return: filled params dict
        """
        for k, v in default_params.items():
            if k not in params:
                params[k] = v

        return params

    def make_initial_params(self, **kwargs):
        raise NotImplementedError

    def make_dataset_params(self, **kwargs):
        raise NotImplementedError

    def make_model_params(self, **kwargs):
        raise NotImplementedError

    def make_train_params(self, **kwargs):
        raise NotImplementedError

    def make_learning_rates(self, **kwargs):
        raise NotImplementedError

    def make_references(self, **kwargs):
        raise NotImplementedError

    def make_model(self, **kwargs) -> CompartmentalModel:
        raise NotImplementedError

    def set_initial_params(self, initial_params):
        self.initial_params = initial_params

    def set_dataset_params(self, dataset_params):
        self.dataset_params = dataset_params

    def set_model_params(self, model_params):
        self.model_params = {**model_params}

    def set_learning_rates(self, learning_rates):
        self.learning_rates = learning_rates

    def set_train_params(self, train_params):
        self.train_params = train_params

    def set_references(self, references):
        self.references = references

    def set_model(self, model):
        self.model = model