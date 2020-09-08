import os
from collections import namedtuple

Country = namedtuple("Country", ["df_file", "area", "area_col_name"])
cwd = os.getcwd()


def get_region_params(name):
    """
    Given the name of a region, it returns the paths and params
    necessary to retrieve the data related to such region name.
    :param name: str, region name, if name not in REGION_NAME_DICT it is assumed to be an italian region
    :return: A named tuple with paths and params to get data.
    """

    REGION_NAME_DICT = {
        "Italy": Country(os.path.join(cwd, "COVID-19", "dati-andamento-nazionale", "dpc-covid19-ita-andamento-nazionale.csv"), ["ITA"], "stato"),
        "UK": Country(os.path.join(cwd, "dati-uk", "uk_data_filled.csv"), ["UK"], "stato"),
        "FR": Country(os.path.join(cwd, "dati-fr", "fr_data_processed.csv"), ["FR"], "stato"),
        "it-region": Country(os.path.join(cwd, "COVID-19", "dati-regioni", "dpc-covid19-ita-regioni.csv"), [name], "denominazione_regione"),
    }

    return REGION_NAME_DICT[name] if name in REGION_NAME_DICT else REGION_NAME_DICT["it-region"]
