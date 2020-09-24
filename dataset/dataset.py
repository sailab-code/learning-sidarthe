import numpy as np

from dataset.config import get_region_params


class Dataset:
    def __init__(self, dataset_params):
        self.region = dataset_params["region"]
        self.train_size = dataset_params["train_size"]
        self.val_len = dataset_params["val_len"]

        self.x, self.y = None, None
        self.first_date = None

    def get_targets(self):
        return self.y

    def load_data(self, region_params):
        return NotImplementedError

    def select_targets(self, targets):
        """
        Removes the initial days until there are 0 cases.
        :param targets: dictionary with targets
        :return: (targets, first_date)
        """

        return NotImplementedError

    @staticmethod
    def normalize_values(values, norm):
        """normalize values by a norm, e.g. population"""
        return {key: np.array(value) / norm for key, value in values.items()}

    def make_dataset(self):
        """
        simply load and prepare the data.
        :return:
        """
        region_params = get_region_params(self.region)
        self.x, self.y = self.load_data(region_params)
        self.x, self.y, self.first_date = self.select_targets(self.y)

    @property
    def inputs(self):
        return self.x

    @property
    def targets(self):
        return self.y
