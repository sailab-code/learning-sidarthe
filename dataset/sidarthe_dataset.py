from dataset.dataset import Dataset

from utils.data_utils import select_data


class SidartheDataset(Dataset):
    def __init__(self, dataset_params):
        super().__init__(dataset_params)

    def load_data(self, region_params):
        groupby_cols = ["data"]  # ["Date"]

        d_col_name = "isolamento_domiciliare"
        r_col_name = "ricoverati_con_sintomi"
        t_col_name = "terapia_intensiva"
        h_detected_col_name = "dimessi_guariti"
        e_col_name = "deceduti"  # "Fatalities"

        df_file, area, area_col_name = region_params.df_file, region_params.area, region_params.area_col_name
        x_target, d_target, dates = select_data(df_file, area, area_col_name, d_col_name, groupby_cols, file_sep=",")
        _, y_target, _ = select_data(df_file, area, area_col_name, "totale_positivi", groupby_cols, file_sep=",")
        _, r_target, _ = select_data(df_file, area, area_col_name, r_col_name, groupby_cols, file_sep=",")
        _, t_target, _ = select_data(df_file, area, area_col_name, t_col_name, groupby_cols, file_sep=",")
        _, h_detected_target, _ = select_data(df_file, area, area_col_name, h_detected_col_name, groupby_cols, file_sep=",")
        _, e_target, _ = select_data(df_file, area, area_col_name, e_col_name, groupby_cols, file_sep=",")

        return x_target, {"y": y_target, "d": d_target, "r": r_target, "t": t_target, "h_detected": h_detected_target, "e": e_target, "dates": dates}

    def select_targets(self, targets):
        y_target, dates = targets["y"], targets["dates"]

        d_target, r_target, t_target, h_detected_target, e_target = targets["d"], targets["r"], targets["t"], targets["h_detected"], targets["e"]
        initial_len = len(y_target)
        tmp_d, tmp_r, tmp_t, tmp_h, tmp_e = [], [], [], [], []
        first_date = None
        for i in range(initial_len):
            # if y_target[i] > 0:
            if d_target[i] + r_target[i] > 0:
                tmp_d = d_target[i:]
                tmp_r = r_target[i:]
                tmp_t = t_target[i:]
                tmp_h = h_detected_target[i:]
                tmp_e = e_target[i:]
                first_date = dates[i]
                break

        d_target = tmp_d
        r_target = tmp_r
        t_target = tmp_t
        h_detected_target = tmp_h
        e_target = tmp_e

        self.y = {
            "d": d_target,
            "r": r_target,
            "t": t_target,
            "h_detected": h_detected_target,
            "e": e_target
        }

        return self.x[-len(self.y["d"]):], self.y, first_date
