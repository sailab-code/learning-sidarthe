from git_manager import GitManager
import utils.data_utils as du
import os.path as op
import os

if __name__ == '__main__':
    git_manager = GitManager()
    filename = op.join(os.curdir, "COVID-19", "dati-regioni", "dpc-covid19-ita-regioni.csv")
    x, y = du.select_data(file=filename,
                          areas=["Toscana"],
                          area_col_name="denominazione_regione",
                          value_col_name=["totale_ospedalizzati",
                                          "totale_attualmente_positivi", "dimessi_guariti",
                                          "deceduti", "totale_casi"]
                          )

    print(f"x={x}, y={y}")
