# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 13:23:41 2020

@author: HO18971
"""

import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy import interpolate
import matplotlib.pyplot as pl
from datetime import datetime as dt
import datetime



#######
# GOOGLE
#######

df_google = pd.read_csv(os.path.join(os.getcwd(),'Global_Mobility_Report.csv'))
df_google['mobility'] = (df_google['retail_and_recreation_percent_change_from_baseline'] + df_google['grocery_and_pharmacy_percent_change_from_baseline'] +df_google['transit_stations_percent_change_from_baseline'] +df_google['workplaces_percent_change_from_baseline'])/4
locations = ['Lombardy', 'France', 'Italy']
df_google = df_google[((df_google['country_region'].isin(locations)) & (df_google['sub_region_1'].isnull())) | (df_google['sub_region_1'].isin(locations))]
df_google = df_google[['sub_region_1', 'country_region', 'mobility', 'date']]
df_google['sub_region_1'] = df_google['sub_region_1'].fillna(df_google['country_region'])
df_google = pd.pivot_table(df_google, index=['date'], values='mobility', columns='sub_region_1')
df_google.index = pd.to_datetime(df_google.index)
df_google.plot(title='Google')
df_google = 1 + df_google/100.

df_google = df_google['Italy']




