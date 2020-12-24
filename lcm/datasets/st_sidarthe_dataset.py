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
        all_dates = []
        y_targets, d_targets, r_targets, t_targets, h_targets, e_targets = [], [], [], [], [], []
        df_file, areas, area_col_name = self.data_path, self.region, self.region_column  # > 1 regions in self.region

        self.n_areas = len(areas)
        for area in areas:
            area = [area]
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
            all_dates.append(dates)

        targets = {
            "y": y_targets,
            "d": d_targets,
            "r": r_targets,
            "t": t_targets,
            "h_detected": h_targets,
            "e": e_targets,
        }

        filtered_targets, first_dates, outbreak_starts, outbreak_max_len = self.select_targets(targets, all_dates)
        return x_targets, filtered_targets, first_dates, outbreak_starts, outbreak_max_len

    @staticmethod
    def select_targets(targets, dates):
        y_target = targets["y"]

        d_target, r_target = targets["d"], targets["r"]
        n_regions, outbreak_max_len = len(y_target), len(y_target[0])  # fixme outbreak max len not right

        first_dates = []
        # finds WHEN outbreak actually starts in each area
        outbreak_start = []
        for i in range(n_regions):
            for j in range(outbreak_max_len):
                if d_target[i][j] + r_target[i][j] > 0:
                    outbreak_start.append(j)
                    first_dates.append(dates[i][j])
                    break

        filtered_targets = {}
        # filter out initial empty days
        for target_key, target_val in targets.items():
            region_target = []
            for i in range(n_regions):
                padded_target = np.concatenate((target_val[i][outbreak_start[i]:], np.array([-1]*outbreak_start[i])))
                region_target.append(padded_target)

            filtered_targets[target_key] = np.array(region_target)
        filtered_targets.pop("y", None) # removes unused key

        outbreak_max_len = outbreak_max_len - min(outbreak_start)
        return filtered_targets, first_dates, np.array(outbreak_start), outbreak_max_len

    def setup(self, stage: Optional[str] = None):
        """
        Setup train/val/test data accordingly to the current stage.
        Each set contains a tuple (x, y, mask), where mask is necessary to isolate the correct samples.

        :param stage: training stage: 'fit' | 'test' | None. 
        :return:
        """

        self.x, self.y, self.first_date, outbreak_starts, outbreak_max_len = self.load_data()

        # Assuring all the regions share the same validation and test intervals
        # This implies that the number of training days may change and it must be PADDED in the end
        # A bit tricky

        range_matrix = np.arange(outbreak_max_len).reshape(1, -1).repeat(self.n_areas, axis=0)

        train_breadth = self.train_size - outbreak_starts  # S
        repeated_train_breadth = train_breadth.reshape(-1, 1).repeat(outbreak_max_len, axis=1)
        after_train_mask = np.greater_equal(range_matrix, repeated_train_breadth) # S x all_size, elements after train are True

        # creates mask for validation elements
        val_breadth = self.val_size + train_breadth
        repeated_val_breadth = val_breadth.reshape(-1, 1).repeat(outbreak_max_len, axis=1)
        after_val_mask = np.greater_equal(range_matrix, repeated_val_breadth)  # elements after validation are True
        before_val_mask = np.bitwise_not(after_val_mask) # elements before validation are True
        val_mask = np.bitwise_and(before_val_mask, after_train_mask)  # elements in validation are True

        # create mask for test elements
        pad_breadth = outbreak_max_len - outbreak_starts  # needed to remove padded elements at the end
        repeated_pad_breadth = pad_breadth.reshape(-1, 1).repeat(outbreak_max_len, axis=1)
        before_pad_mask = np.less(range_matrix, repeated_pad_breadth)  # elements before pad are True
        test_mask = np.bitwise_and(after_val_mask, before_pad_mask) # elements in test are true
        self.test_size = outbreak_max_len - self.train_size - self.val_size

        train_set, val_set, test_set = {}, {}, {}
        for target_key, target_value in self.y.items():
            # creates train data
            train_y = np.copy(target_value)
            train_y[after_train_mask] = -1  # padding values are needed in train
            train_y = torch.tensor(train_y).transpose(0,1)  # shape becomes T x S (because more compliant for the rest of the code)
            train_y = train_y[:self.train_size,:]
            train_set[target_key] = train_y[:self.train_size,:] # keep only up to train size

            # creates val data
            val_y = np.copy(target_value[val_mask].reshape(self.n_areas, self.val_size))
            val_y = torch.tensor(val_y, dtype=torch.float32).transpose(0,1)  # shape becomes T x S (because more compliant for the rest of the code)

            masked_train_y = -torch.ones(self.train_size, self.n_areas)
            val_set[target_key] = torch.cat((masked_train_y, val_y), dim=0)  # make target train+val shaped filled with -1 in train

            # creates test data
            test_y = np.copy(target_value[test_mask].reshape(self.n_areas, self.test_size))
            test_y = torch.tensor(test_y, dtype=torch.float32).transpose(0,1)  # shape becomes T x S (because more compliant for the rest of the code)
            masked_train_val_y = -torch.ones(self.train_size+self.val_size, self.n_areas)
            test_set[target_key] = torch.cat((masked_train_val_y, test_y), dim=0) # make target train+val+test shaped filled with -1 in train+val

        train_slice = slice(0, self.train_size, 1)
        train_val_slice = slice(0, self.train_size + self.val_size)
        val_slice = slice(self.train_size, self.train_size + self.val_size)
        test_slice = slice(self.train_size + self.val_size, outbreak_max_len)
        all_slice = slice(0, outbreak_max_len)

        # t_grid = torch.tensor(self.x).transpose(0,1) # shape becomes T x S (because more compliant for the rest of the code)
        t_grid = torch.range(0, outbreak_max_len).reshape(-1,1) # shape becomes T x S (because more compliant for the rest of the code)
        if stage == 'fit' or stage is None:
            train_mask = torch.ones_like(t_grid[train_slice, :]).type(torch.bool)
            self.train_set = DictDataset([(t_grid[train_slice, :], train_set, train_mask)])

            val_mask = torch.zeros_like(t_grid[train_val_slice,:])
            val_mask[val_slice] = 1.0
            val_mask = val_mask.type(torch.bool)
            self.val_set = DictDataset([(t_grid[train_val_slice, :], val_set, val_mask)])

        if stage == 'test' or stage is None:
            test_mask = torch.zeros_like(t_grid[all_slice, :])
            test_mask[test_slice] = 1.0
            test_mask = test_mask.type(torch.bool)
            self.test_set = DictDataset([(t_grid[all_slice, :], test_set, test_mask)])
            self.test_size = outbreak_max_len - self.train_size - self.val_size



    def get_initial_conditions(self, population):
        """
        Compute initial conditions from initial target values (S x 8)
        targets = {
            "d": "isolamento_domiciliare",
            "r": "ricoverati_con_sintomi",
            "t": "terapia_intensiva",
            "h_detected": "dimessi_guariti",
            "e": "deceduti"
        }
        """
        targets = self.train_set.target_dicts[0][1]
        D0 = targets["d"][0, :]  # isolamento
        R0 = targets["r"][0, :]  # ricoverati con sintomi
        T0 = targets["t"][0, :]  # terapia intensiva
        H0_detected = targets["h_detected"][0, :]  # dimessi guariti
        E0 = targets["e"][0, :]  # deceduti

        # for now we assume that the number of undetected is equal to the number of detected
        # meaning that half of the infectious were not detected
        I0 = D0  # isolamento domiciliare
        A0 = R0  # ricoverati con sintomi
        population = torch.tensor(population)
        S0 = population - (I0 + D0 + A0 + R0 + T0 + H0_detected + E0)

        initial_states = (
            S0,
            I0,
            D0,
            A0,
            R0,
            T0,
            E0,
            H0_detected
        )

        return torch.cat(initial_states, dim=0).reshape(1, -1, len(initial_states))
