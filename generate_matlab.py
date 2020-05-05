import pandas as pd
import os

file = os.path.join(os.getcwd(), "COVID-19", "dati-andamento-nazionale", "dpc-covid19-ita-andamento-nazionale.csv")
file_sep = ","

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


value_col_name = ["totale_casi",
                  "deceduti",
                  "dimessi_guariti",
                  "totale_positivi",
                  "isolamento_domiciliare",
                  "ricoverati_con_sintomi",
                  "terapia_intensiva"
                  ]
groupby_cols = ["data"]

df = pd.read_csv(file, sep=file_sep)
y = select_column_values(df, col_name=value_col_name, groupby=groupby_cols)
x = list(range(1, len(y)+1))

def printMatlabVar(name, vals, prefix = []):
    totVals = prefix + list(vals)
    print(f"{name} = [{' '.join(map(str, totVals))}]/popolazione")

printMatlabVar("CasiTotali", y[:,0], [3, 20, 79, 132])
printMatlabVar("Deceduti", y[:,1], [0, 1, 2, 2])
printMatlabVar("Guariti", y[:,2], [0, 0, 0, 1])
printMatlabVar("Positivi", y[:,3], [3, 19, 77, 129])
printMatlabVar("Isolamento_domiciliare", y[:,4], [49])
printMatlabVar("Ricoverati_sintomi", y[:,5], [54])
printMatlabVar("Terapia_intensiva", y[:,6], [26])



