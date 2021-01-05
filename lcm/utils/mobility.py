import os
import pandas as pd
import torch
#######
# GOOGLE
#######

mobility_map = {
    "Lombardia": "Lombardy",
    "Toscana": "Tuscany"
}


def get_google_mobility(locations, first_date):
    """

    :param locations: a list of regions e.g. ['Lombardy', 'France', 'Italy']
    :param first_date: a string with the first date
    :return: a dataframe containing such locations
    """

    locations = [mobility_map[location] if location in mobility_map else location for location in locations]

    df_google = pd.read_csv(os.path.join(os.getcwd(), 'data', 'Global_Mobility_Report.csv'))
    df_google['mobility'] = (df_google['retail_and_recreation_percent_change_from_baseline'] + df_google['grocery_and_pharmacy_percent_change_from_baseline'] +df_google['transit_stations_percent_change_from_baseline'] +df_google['workplaces_percent_change_from_baseline'])/4
    df_google = df_google[((df_google['country_region'].isin(locations)) & (df_google['sub_region_1'].isnull())) | (df_google['sub_region_1'].isin(locations))]
    df_google = df_google[['sub_region_1', 'country_region', 'mobility', 'date']]
    df_google['sub_region_1'] = df_google['sub_region_1'].fillna(df_google['country_region'])
    df_google = pd.pivot_table(df_google, index=['date'], values='mobility', columns='sub_region_1')
    df_google.index = pd.to_datetime(df_google.index)
    df_google.plot(title='Google')
    df_google = 1 + df_google/100.

    # df_google = df_google['Italy']
    df_google = df_google.loc[first_date[0].split("T")[0]:, :]
    df_google = torch.tensor(df_google.values, requires_grad=False, dtype=torch.float32)
    return df_google # fixme va sistemato per regione