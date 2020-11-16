import pandas as pd


def select_regions(df, regions, col_name="denominazione_regione"):
    """
    Select rows by values in regions from column col_name
    :param df: pandas dataFrame
    :param regions: a list of values
    :param col_name: a string indicating the column
    :return: a new DataFrame with only the selected rows of df
    """
    return df[df[col_name].isin(regions)].reset_index()


def select_column_values(df, col_name="totale_casi", groupby=["data"], group_by_criterion="sum"):
    """

    :param df: pandas dataFrame
    :param col_name: column of interest
    :param groupby: column to group by (optional) data.
    :param group_by_criterion: how to merge the values of grouped elements in col_name.
    Only sum supported.
    :return: a list of of values
    """
    if groupby is not None:
        if group_by_criterion == "sum":
            return df.groupby(by=groupby)[col_name].sum().reset_index()[col_name].values
        else:
            return RuntimeWarning
    else:
        return list(df[col_name])


def select_data(file, areas, area_col_name, value_col_name, groupby_cols, file_sep=","):
    """
    Function to load any csv file, selecting which rows to get and which column
    :param file: location of csv file with the data
    :param areas: a list of values (strings) of the areas to select
    :param area_col_name: name (string) of the column related to the 'area' field.
    :param value_col_name: name (string) of the column with the values of interest
    :param groupby_cols: columns to groupby
    :param file_sep: separator of the csv file, (default) ','
    :return: x,y and dates, where x is just a range of integers from 1 (day 0) to N (last day), y are the
    values of the column selected, dates are the dates from day 0 to last day.

    Example of usage:
        Getting time series of deaths in Toscana:

            df_file = os.path.join(os.getcwd(), "dati-regioni", "dpc-covid19-ita-regioni.csv")
            areas = ["Toscana"]
            area_col_name = "denominazione_regione"
            value_col_name = "deceduti"
            x,w = get_data(df_file, areas, area_col_name, value_col_name, file_sep=",")

        Getting time series of deaths in whole Italy:
            just the same except for:
            areas = list(df["denominazione_regione"].unique())
    """

    df = pd.read_csv(file, sep=file_sep)
    df = df.fillna(-1)  # set nans to -1
    area_df = select_regions(df, areas, col_name=area_col_name)
    y = select_column_values(area_df, col_name=value_col_name, groupby=groupby_cols)
    x = list(range(1, len(y)+1))
    dates = select_column_values(area_df, col_name="data", groupby=None)

    return x, y, dates
