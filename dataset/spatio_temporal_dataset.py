import numpy as np

from dataset.sidarthe_dataset import SidartheDataset
from utils.data_utils import select_data


class SpatioTemporalSidartheDataset(SidartheDataset):
    def __init__(self, dataset_params):
        super().__init__(dataset_params)
        self.batch_size = dataset_params["n_areas"]

        # TODO create load data
        # TODO careful with first dates, must be an array
        # TODO DIFFERENT INITIAL CONDITIONS REQUIRED

    def load_data(self, region_params):
        groupby_cols = ["data"]  # ["Date"]

        d_col_name = "isolamento_domiciliare"
        r_col_name = "ricoverati_con_sintomi"
        t_col_name = "terapia_intensiva"
        h_detected_col_name = "dimessi_guariti"
        e_col_name = "deceduti"  # "Fatalities"

        x_targets = []
        all_dates = []
        y_targets, d_targets, r_targets, t_targets, h_targets, e_targets = [], [], [], [], [], []
        df_file, areas, area_col_name = region_params.df_file, region_params.area, region_params.area_col_name
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
            all_dates.append(dates)


        return x_targets, {"y": y_targets, "d": d_targets, "r": r_targets, "t": t_targets, "h_detected": h_targets, "e": e_targets, "dates": all_dates}

    def select_targets(self, targets):
        y_target, dates = targets["y"], targets["dates"]

        d_target, r_target, t_target, h_detected_target, e_target = targets["d"], targets["r"], targets["t"], targets["h_detected"], targets["e"]
        n_regions, initial_len = len(y_target), len(y_target[0])
        first_dates = []

        # finds when outbreak actually starts in each region
        outbreak_start = []
        for i in range(n_regions):
            for j in range(initial_len):
                # if y_target[i] > 0:
                if d_target[i][j] + r_target[i][j] > 0:
                    outbreak_start.append(j)
                    first_dates.append(dates[i][j])
                    break

        # filter out initial empty days
        outbreak_lengths = [initial_len - o for o in outbreak_start]
        filtered_targets = {"mask": outbreak_lengths} # storing outbreak length for each area, is a list of size S
        for target_key, target_val in targets.items():
            region_target = []
            for i in range(n_regions):
                region_target.append(target_val[i][outbreak_start[i]:])

            filtered_targets[target_key] = region_target


        # pos_matrix = np.arange(initial_len).reshape(1,-1).repeat(n_regions, axis=0)  # S x T ranges
        # outbreak_start_mask = np.array(outbreak_start).reshape(-1,1).repeat(initial_len)  # S x T start mask
        # np.greater_equal(pos_matrix, outbreak_start_mask)

        self.y = filtered_targets
        return self.x, self.y, first_dates
