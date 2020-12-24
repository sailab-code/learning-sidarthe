import abc
from typing import Union, List, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class DictDataset(Dataset):

    def __init__(self, target_dicts):
        self.target_dicts = target_dicts

    def __getitem__(self, index: int):
        return self.target_dicts[index]

    def __len__(self) -> int:
        return len(self.target_dicts)


class ODEDataModule(pl.LightningDataModule):

    def __init__(self, region, data_path, train_size, val_size, region_column="stato"):
        super().__init__()
        self.region = region
        self.region_column = region_column
        self.data_path = data_path
        self.x, self.y = None, None
        self.first_date = None

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = None

        self.train_set = None
        self.val_set = None
        self.test_set = None

    @abc.abstractmethod
    def load_data(self):
        pass

    @staticmethod
    def slice_targets(targets, sl):
        return {key: torch.tensor(value[sl]) for key, value in targets.items()}

    def setup(self, stage: Optional[str] = None):
        self.x, self.y, self.first_date = self.load_data()

        train_slice = slice(0, self.train_size, 1)
        train_val_slice = slice(0, self.train_size+self.val_size)
        val_slice = slice(self.train_size, self.train_size + self.val_size)
        test_slice = slice(self.train_size + self.val_size, len(self.x))
        all_slice = slice(0, len(self.x))

        t_grid = torch.tensor(self.x)

        if stage == 'fit' or stage is None:
            train_mask = torch.ones_like(t_grid[train_slice]).type(torch.bool)
            self.train_set = DictDataset([(t_grid[train_slice], self.slice_targets(self.y, train_slice), train_mask)])

            val_mask = torch.zeros_like(t_grid[train_val_slice])
            val_mask[val_slice] = 1.0
            val_mask = val_mask.type(torch.bool)

            self.val_set = DictDataset([(t_grid[train_val_slice], self.slice_targets(self.y, train_val_slice), val_mask)])

        if stage == 'test' or stage is None:
            test_mask = torch.zeros_like(t_grid[all_slice])
            test_mask[test_slice] = 1.0
            test_mask = test_mask.type(torch.bool)
            self.test_set = DictDataset([(t_grid[all_slice], self.slice_targets(self.y, all_slice), test_mask)])
            self.test_size = len(self.x) - self.train_size - self.val_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=1)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_set, batch_size=1)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_set, batch_size=1)

    @abc.abstractmethod
    def get_initial_conditions(self, population):
        """
        Returns initial conditions for the compartmental model
        :param population: number of individuals in the population.
        It can be a scalar or a list, depending on the number of populations
        considered.
        :return: A tensor with the initial state.
        """
        pass

