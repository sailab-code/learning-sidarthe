from lcm.datasets import ODEDataModule
from ..utils.data import select_data


class SirDataModule(ODEDataModule):
    def load_data(self):
        groupby_cols = ["data"]  # ["Date"]

        infected_col_name = "totale_positivi"
        healed_col_name = "dimessi_guariti"  # healed
        death_col_name = "deceduti"  # death

        df_file, area, area_col_name = self.data_path, self.region, self.region_column

        # I = infected
        x_target, i_target, dates = select_data(df_file, area, area_col_name, infected_col_name, groupby_cols, file_sep=",")

        # R = deaths + healed
        _, h_target, _ = select_data(df_file, area, area_col_name, healed_col_name, groupby_cols, file_sep=",")
        _, d_target, _ = select_data(df_file, area, area_col_name, death_col_name, groupby_cols, file_sep=",")

        # removing days before outbreak begins
        initial_len = len(i_target)
        tmp_i, tmp_r = [], []
        first_date = None
        for i in range(initial_len):
            if i_target[i] > 0:
                tmp_i = i_target[i:]
                tmp_r = h_target[i:] + d_target[i:]
                first_date = dates[i]
                break

        i_target = tmp_i
        r_target = tmp_r

        self.x = x_target

        self.y = {
            "i": i_target,
            "r": r_target,
        }
        return self.x[-len(self.y["i"]):], self.y, first_date
