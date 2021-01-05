import os
import pandas as pd
import torch

#######
# GOOGLE
#######

mobility_map = {
    "Italia": "Italy",
    "Lombardia": "Lombardy",
    "Toscana": "Tuscany"
}


def get_google_mobility(locations, first_date):
    """

    :param locations: a list of regions e.g. ['Lombardy', 'France', 'Italy']
    :param first_date: a string with the first date
    :return: a dataframe containing such locations
    """

    # include only used columns to save memory and time
    percent_change_columns = [
        f"{key}_percent_change_from_baseline"
        for key in [
            "retail_and_recreation",
            "grocery_and_pharmacy",
            "transit_stations",
            "workplaces"
        ]
    ]

    used_cols = [
        *percent_change_columns,
        "country_region",
        "sub_region_1",
        "date"
    ]

    df_google = pd.read_csv(os.path.join(os.getcwd(), 'data', 'Global_Mobility_Report.csv'), usecols=used_cols)

    # sum percent changes
    df_google['mobility'] = sum([
        df_google[key] for key in percent_change_columns]
    ) / 4

    # filter out locations
    locations = [mobility_map[location] if location in mobility_map else location for location in locations]
    country_region_in_locations = df_google['country_region'].isin(locations)
    is_subregion1_null = df_google['sub_region_1'].isnull()
    subregion1_in_locations = df_google['sub_region_1'].isin(locations)
    df_google = df_google[
        (country_region_in_locations & is_subregion1_null) | subregion1_in_locations
        ]

    df_google = df_google[['sub_region_1', 'country_region', 'mobility', 'date']]
    df_google['sub_region_1'] = df_google['sub_region_1'].fillna(df_google['country_region'])
    df_google = pd.pivot_table(df_google, index=['date'], values='mobility', columns='sub_region_1')
    df_google.index = pd.to_datetime(df_google.index)
    df_google.plot(title='Google')
    df_google = 1 + df_google / 100.

    # df_google = df_google['Italy']
    # filter by first date onward
    df_google = df_google.loc[first_date[0].split("T")[0]:, :]
    df_google = torch.tensor(df_google.values, requires_grad=False, dtype=torch.float32)
    return df_google  # fixme va sistemato per regione
