# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 2024

@author: Georg Erharter / Paul Unterlass

File containing several helper functions
"""


# script with utilities, formulas etc

import numpy as np
from os import listdir
import pandas as pd
from scipy import interpolate

class utilities:

    def __init__(self):
        pass
    
    
    def equally_spaced_df(self, df, tunnellength, interval):
        min_length = df[tunnellength].min()
        max_length = df[tunnellength].max()

        equal_range = np.arange(min_length, max_length, interval)

        df_interp = pd.DataFrame({tunnellength: equal_range})

        for feature in df.drop(tunnellength, axis=1).columns:
            f = interpolate.interp1d(df[tunnellength], df[feature],
                                     kind='linear')
            df_interp[feature] = f(equal_range)

        df_interp.set_index(tunnellength, drop=False, inplace=True)

        return df_interp

    def concat_tables_BBT(self, folder, drop_standstills=True,
                          check_for_miss_vals=True):

        filenames = []
        for file in listdir(folder):
            if file.split('.')[1] == 'csv':
                filenames.append(file)

        files = []

        for i, file in enumerate(filenames[:1000]):

            df = pd.read_csv(fr'{folder}\{file}', sep=';') # , skiprows=3, parse_dates=['Date']
            print(file)
            
            df = df.drop(df.columns[0], axis=1)
            column_names = ['Date', 'Stroke', 'Chainage [m]', 'Tunnel Distance [m]',
                'Advance speed [mm/min]', 'Speed cutterhead [rpm]',
                'Pressure advance cylinder bottom side [bar]',
                'Torque cutterhead [MNm]',
                'Total advance force [kN]', 'Penetration [mm/rot]',
                'Pressure RSC left [bar]',
                'Pressure RSC right[bar]',
                'Path RSC left [mm]']

            df.columns = column_names
            
            # getting rid of datapoints < 0 in Tunnel Distance or nan
            df = df[(df['Tunnel Distance [m]'] >=0 | (df['Tunnel Distance [m]'].isnull()))]

            if drop_standstills is True:
                # drop obvious standstills
                idx_standstill = df.loc[(df['Torque cutterhead [MNm]'] <= 0)
                                        | (df['Total advance force [kN]'] <= 0)
                                        | (df['Speed cutterhead [rpm]'] <= 0)
                                        | (df['Penetration [mm/rot]'] <= 0)].index
                df.drop(idx_standstill, inplace=True)

            files.append(df)
            print(f'{i} / {len(filenames)-1} csv done')  # status

        df = pd.concat(files, sort=True)
        df.dropna(inplace=True)
        
        # check for missing values in time series
        if check_for_miss_vals is True:
            self.check_for_miss_vals(df)
        
        return df, files

    def concat_tables_UT(self, folder, drop_standstills=True,
                      check_for_miss_vals=True):

        filenames = []
        for file in listdir(folder):
            if file.split('.')[1] == 'csv':
                filenames.append(file)

        files = []

        for i, file in enumerate(filenames[:1000]):

            df = pd.read_csv(fr'{folder}\{file}', sep=';', skiprows=3, parse_dates=['Date'])
            print(file)

            # hard drop of most obvious outliers
            df.drop(df[df['Tunnel Distance [m]'] > 7000].index, inplace=True)

            if drop_standstills is True:
                # drop obvious standstills
                idx_standstill = df.loc[(df['Torque cutterhead [MNm]'] <= 0)
                                        | (df['Total advance force [kN]'] <= 0)
                                        | (df['Speed cutterhead for display [rpm]'] <= 0)
                                        | (df['Penetration [mm/rot]'] <= 0)].index
                df.drop(idx_standstill, inplace=True)

            files.append(df)
            print(f'{i} / {len(filenames)-1} csv done')  # status

        df = pd.concat(files, sort=True)
        print('# datapoints without standstills', df['Tunnel Distance [m]'].count())
        #try:
        #   df.drop(['Tunnel length [m]'], axis=1, inplace=True)
        #except KeyError:
        #    pass
        #df.dropna(inplace=True)
        

        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values(by='Date', inplace=True)

        # check for missing values in time series
        if drop_standstills is False and check_for_miss_vals is True:
            self.check_for_miss_vals(df)

        return df, files
    
    def concat_tables_FB(self, folder, drop_standstills=True,
                         check_for_miss_vals=True):

        filenames = []
        for file in listdir(folder):
            if file.split('.')[1] == 'csv':
                filenames.append(file)

        files = []

        for i, file in enumerate(filenames[:5000]):

            df = pd.read_csv(fr'{folder}\{file}', sep=';', parse_dates=['Timestamp'])
            print(file)

            # convert strings to numeric in order to get NaNs instead of strings (e.g., \N)
            df = df.apply(pd.to_numeric, errors='coerce')

            if drop_standstills is True:
                # drop obvious standstills
                idx_standstill = df.loc[(df['CH Torque [MNm]'] <= 0)
                                        | (df['Thrust Force [kN]'] <= 0)
                                        | (df['CH Rotation [rpm]'] <= 0)
                                        | (df['CH Penetration [mm/rot]'] <= 0)].index
                df.drop(idx_standstill, inplace=True)

            files.append(df)
            print(f'{i} / {len(filenames)-1} csv done')  # status

        df = pd.concat(files, sort=True)        

        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.sort_values(by='Timestamp', inplace=True)

        # check for missing values in time series
        if drop_standstills is False and check_for_miss_vals is True:
            self.check_for_miss_vals(df)

        return df