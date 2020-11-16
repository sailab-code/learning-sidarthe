import numpy as np
import torch
from typing import Optional

from lcm.datasets.sidarthe_dataset import SidartheDataModule
from lcm.datasets import DictDataset
from utils.data_utils import select_data


class SpatioTemporalSidartheDataset(SidartheDataModule):
    def __init__(self, region, data_path, train_size, val_size, region_column="stato"):
        super().__init__(region, data_path, train_size, val_size, region_column=region_column)
        self.n_areas = None

    def load_data(self):
        groupby_cols = ["data"]  # ["Date"]

        d_col_name = "isolamento_domiciliare"
        r_col_name = "ricoverati_con_sintomi"
        t_col_name = "terapia_intensiva"
        h_detected_col_name = "dimessi_guariti"
        e_col_name = "deceduti"  # "Fatalities"

        x_targets = []
        y_targets, d_targets, r_targets, t_targets, h_targets, e_targets = [], [], [], [], [], []
        df_file, areas, area_col_name = self.data_path, self.region, self.region_column  # > 1 regions in self.region
        self.n_areas = len(areas)
        for area in areas:
            x_target, d_target, dates = select_data(df_file, area, area_col_name, d_col_name, groupby_cols, file_sep=",")
            _, y_target, _ = select_data(df_file, area, area_col_name, "totale_positivi", groupby_cols, file_sep=",")
            _, r_target, _ = select_data(df_file, area, area_col_name, r_col_name, groupby_cols, file_sep=",")
            _, t_target, _ = select_data(df_file, area, area_col_name, t_col_name, groupby_cols, file_sep=",")
            _, h_detected_target, _ = select_data(df_file, area, area_col_name, h_detected_col_name, groupby_cols, file_sep=",")
            _, e_target, _ = select_data(df_file, area, area_col_name, e_col_name, groupby_cols, file_sep=",")
            x_targets.append(x_target)
            y_targets.append(y_target)
            d_targets.append(d_target)
            r_targets.append(r_target)
            t_targets.append(t_target)
            h_targets.append(h_detected_target)
            e_targets.append(e_target)

        targets = {
            "y": y_targets,
            "d": d_targets,
            "r": r_targets,
            "t": t_targets,
            "h_detected": h_targets,
            "e": e_targets,
        }

        filtered_targets, first_dates, outbreak_starts = self.select_targets(targets)
        return x_targets, filtered_targets, first_dates, outbreak_starts

    @staticmethod
    def select_targets(targets):
        y_target, dates = targets["y"], targets["dates"]

        d_target, r_target = targets["d"], targets["r"]
        n_regions, initial_len = len(y_target), len(y_target[0])
        first_dates = []

        # finds WHEN outbreak actually starts in each area
        outbreak_start = []
        for i in range(n_regions):
            for j in range(initial_len):
                if d_target[i][j] + r_target[i][j] > 0:
                    outbreak_start.append(j)
                    first_dates.append(dates[i][j])
                    break

        filtered_targets = {}
        # filter out initial empty days
        for target_key, target_val in targets.items():
            region_target = []
            for i in range(n_regions):
                region_target.append(target_val[i][outbreak_start[i]:])

            filtered_targets[target_key] = np.array(region_target)

        return filtered_targets, first_dates, outbreak_start # outbreak_lengths

    def setup(self, stage: Optional[str] = None):
        self.x, self.y, self.first_date, outbreak_starts = self.load_data()

        # Assuring all the regions share the same validation and test intervals
        # This implies that the number of training days may change
        # A bit tricky but it shall work
        train_outbreak_sizes = self.train_size - outbreak_starts
        train_range_matrix = np.arange(train_outbreak_sizes).reshape(1, -1).repeat(self.n_areas, axis=0)  # S x train_size ranges
        repeated_outbreak_sizes = train_outbreak_sizes.reshape(-1, 1).repeat(train_outbreak_sizes)  # S x train_size lengths
        train_outbreak_len_mask = np.greater_equal(train_range_matrix, repeated_outbreak_sizes) # S x train_size

        val_outbreak_sizes = train_outbreak_sizes + self.val_size
        val_range_matrix = np.arange(train_outbreak_sizes, val_outbreak_sizes).reshape(1, -1).repeat(self.n_areas, axis=0)  # S x val_size ranges
        repeated_outbreak_lengths = val_outbreak_sizes.reshape(-1, 1).repeat(self.val_size)  # S x val_slices
        val_outbreak_len_mask = np.less_equal(val_range_matrix, repeated_outbreak_lengths)  # S x val_size

        test_range_matrix = np.arange(val_outbreak_sizes, val_outbreak_sizes + self.test_size).reshape(1, -1).repeat(self.n_areas, axis=0)  # S x val_size ranges
        repeated_outbreak_lengths = val_outbreak_sizes.reshape(-1, 1).repeat(self.test_size)  # S x test_slices
        test_outbreak_len_mask = np.greater_equal(test_range_matrix, repeated_outbreak_lengths)  # S x test_size

        train_set, val_set, test_set = {}, {}, {}
        for target_key, target_value in self.y.items():
            # creates train data
            train_y = np.copy(target_value[:self.train_size, :])
            train_y[train_outbreak_len_mask] = -1
            train_y = torch.tensor(train_y).transpose(0,1)  # shape becomes T x S (because more compliant for the rest of the code)
            train_set[target_key] = train_y

            # creates val data
            val_y = np.copy(target_value[val_outbreak_len_mask].reshape(self.n_areas, self.val_size))
            val_y = torch.tensor(val_y).transpose(0,1)  # shape becomes T x S (because more compliant for the rest of the code)
            val_set[target_key] = val_y

            # creates test data
            test_y = np.copy(target_value[test_outbreak_len_mask].reshape(self.n_areas, self.test_size))
            test_y = torch.tensor(test_y).transpose(0,1)  # shape becomes T x S (because more compliant for the rest of the code)
            test_set[target_key] = test_y

        train_slice = slice(0, self.train_size, 1)
        val_slice = slice(self.train_size, self.train_size + self.val_size)
        test_slice = slice(self.train_size + self.val_size, len(self.x))

        t_grid = torch.tensor(self.x).transpose(0,1) # shape becomes T x S (because more compliant for the rest of the code)
        if stage == 'fit' or stage is None:
            self.train_set = DictDataset([(t_grid[train_slice, :], train_set)])
            self.val_set = DictDataset([(t_grid[val_slice, :], val_set)])

        if stage == 'test' or stage is None:
            self.test_set = DictDataset([(t_grid[test_slice, :], test_set)])
            self.test_size = len(self.x) - self.train_size - self.val_size
