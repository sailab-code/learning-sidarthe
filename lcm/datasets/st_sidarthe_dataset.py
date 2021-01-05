import numpy as np
import torch
from typing import Optional

from lcm.datasets.sidarthe_dataset import SidartheDataModule
from lcm.datasets import DictDataset
from lcm.utils.data import select_data
from lcm.utils.mobility import get_google_mobility


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
        # This implies that the number of training days may change. PADDING must always be added at the end
        # A bit tricky

        range_matrix = np.arange(outbreak_max_len).reshape(1, -1).repeat(self.n_areas, axis=0)
        train_breadth = self.train_size - outbreak_starts  # S
        repeated_train_breadth = train_breadth.reshape(-1, 1).repeat(outbreak_max_len, axis=1)
        after_train_mask = np.greater_equal(range_matrix, repeated_train_breadth) # S x all_size, elements after train are True
        '''
        E.g. after_train_mask = [
                            [0 0 0 0 1], 
                            [0 0 1 1 1], 
                            [0 0 0 0 1], 
                            [0 0 0 1 1]
                            ...
                            ]
        '''

        # creates mask for validation elements
        val_breadth = self.val_size + train_breadth
        repeated_val_breadth = val_breadth.reshape(-1, 1).repeat(outbreak_max_len, axis=1)
        after_val_mask = np.greater_equal(range_matrix, repeated_val_breadth)  # elements after validation are True
        before_val_mask = np.bitwise_not(after_val_mask) # elements before validation are True
        val_mask = np.bitwise_and(before_val_mask, after_train_mask)  # elements in validation are True
        # adjusting validation mask
        val_mask = torch.tensor(val_mask).transpose(0, 1)  # becomes T x S
        val_mask = val_mask[: (self.train_size + self.val_size), :]

        # create mask for test elements
        pad_breadth = outbreak_max_len - outbreak_starts  # needed to remove padded elements at the end
        repeated_pad_breadth = pad_breadth.reshape(-1, 1).repeat(outbreak_max_len, axis=1)
        before_pad_mask = np.less(range_matrix, repeated_pad_breadth)  # elements before pad are True
        after_pad_mask = np.bitwise_not(before_pad_mask)
        test_mask = np.bitwise_and(after_val_mask, before_pad_mask) # elements in test are true
        # adjusting test mask
        test_mask = torch.tensor(test_mask).transpose(0, 1)  # becomes T x S
        self.test_size = outbreak_max_len - self.train_size - self.val_size

        train_set, val_set, test_set = {}, {}, {}
        for target_key, target_value in self.y.items():
            # creates train data
            train_y = np.copy(target_value)
            train_y[after_train_mask] = -1  # padding values are needed in train
            train_y = torch.tensor(train_y, dtype=torch.float32).transpose(0,1)  # shape becomes T x S (because more compliant for the rest of the code)
            train_y = train_y[:self.train_size,:]
            train_set[target_key] = train_y[:self.train_size,:] # keep only up to train size

            # creates val data
            val_y = np.copy(target_value)
            val_y[after_val_mask] = -1
            val_y = torch.tensor(val_y, dtype=torch.float32).transpose(0, 1) # becomes T x S
            val_set[target_key] = val_y[:(self.train_size + self.val_size), :]  # make target train+val shaped filled with -1 at the end

            # creates test data
            test_y = np.copy(target_value)
            test_y[after_pad_mask] = -1
            test_set[target_key] = torch.tensor(test_y, dtype=torch.float32).transpose(0, 1)

        train_slice = slice(0, self.train_size, 1)
        train_val_slice = slice(0, self.train_size + self.val_size)
        all_slice = slice(0, outbreak_max_len)

        t_grid = torch.arange(0, outbreak_max_len).reshape(-1,1) # shape becomes T x S (because more compliant for the rest of the code)
        if stage == 'fit' or stage is None:
            train_mask = torch.ones_like(t_grid[train_slice, :]).type(torch.bool)
            self.train_set = DictDataset([(t_grid[train_slice, :], train_set, train_mask)])

            self.val_set = DictDataset([(t_grid[train_val_slice, :], val_set, val_mask)])

        if stage == 'test' or stage is None:
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
            S0.reshape(1,-1),
            I0.reshape(1,-1),
            D0.reshape(1,-1),
            A0.reshape(1,-1),
            R0.reshape(1,-1),
            T0.reshape(1,-1),
            E0.reshape(1,-1),
            H0_detected.reshape(1,-1)
        )
        # todo is there a way to avoid reshapes?

        return torch.cat(initial_states, dim=0).transpose(0,1).reshape(1, -1, len(initial_states))


    def get_mobility(self):
        return get_google_mobility(self.region, self.first_date)

