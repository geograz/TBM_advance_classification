# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 2024

@author: Paul Unterlass

Preprocessing functions for the "Gemeinschaftskraftwerk Inn" TBM dataset
"""

from WGAN_00_utilities import utilities
utils = utilities() 

from os import listdir
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pickle import dump


# =============================================================================
# combine single raw data files
# =============================================================================

machine = 'fact_tbm_1'
folder = fr'00_data\01_raw\03_GKI\{machine}'

dfs = []

for i, file in enumerate(listdir(folder)[:33000]):
    if i % 1000 == 0:
        print(f'{i} of {len(listdir(folder))}')
    fp = f'{folder}\{file}'
    #print(fp)
    df = pd.read_parquet(fp)  # , columns=['dim_time', 'dim_station', 'dim_ring'])  # , 'v3'])
    df = df.loc[:, ['dim_time', 'v12', 'v3', 'v17', 'v28', 'v101']]
    dfs.append(df)
    
print('all dataframes loaded -> concatenation')

df_main = pd.concat(dfs)
print('concatenation finished')

print(df_main['dim_time'].duplicated(keep='last').unique())

df_main['dim_time'] = pd.to_datetime(df_main['dim_time'])
df_main = df_main.sort_values(by='dim_time')
df_main = df_main.rename(columns={'v12': 'Stationierung [m]',
                                  'v3': 'Torque cutterhead [kNm]',
                                  'v17': 'Penetration [mm/rot]',
                                  'v101': 'Total advance force [kN]'})

# drop obvious standstills
idx_standstill = df_main.loc[(df_main['Torque cutterhead [kNm]'] <= 0)
                        | (df_main['Total advance force [kN]'] <= 0)
                        | (df_main['Penetration [mm/rot]'] <= 0)].index
df = df_main.drop(idx_standstill, inplace=True)

df.to_parquet(fr'00_data\02_combined\GKI_TBMdata_{machine}.gzip', index=False)

# =============================================================================
# preprocessing pipeline to create training data
# =============================================================================

def preprocessor(df, machine):
    # df = pd.read_parquet(r'00_data\02_combined\GKI_TBMdata_fact_TBM_1.gzip')
    print('\n# datapoints', df['Stationierung [m]'].count())
            
    df = df[(df['Stationierung [m]'] >=0 | (df['Stationierung [m]'].isnull()))]
    df_median = df.groupby('Stationierung [m]', as_index = False).median()
    print('\n# datapoints after grouping by Tunnel Distance', df_median['Stationierung [m]'].count())
    ###########################################################################
    # linear interpolation
    # difference in Tunnel Distance between single data points
    interval = df_median['Stationierung [m]'].diff().median()
    interval = round(interval, 3)
    df_equal = utils.equally_spaced_df(df_median, 'Stationierung [m]', interval)
    
    print('# datapoints after linear interp.',
          df_equal['Stationierung [m]'].count())

    df = df_equal
    df['Stationierung [m] round'] = np.round(df['Stationierung [m]'], 3)
    df.set_index('Stationierung [m] round', inplace=True, drop=False)
    df = df.drop(['Stationierung [m] round', ], axis=1)
    df['Torque cutterhead [MNm]'] = df['Torque cutterhead [kNm]'] / 1000
    df.isna().sum()
    df = df.dropna()
    ###########################################################################
    # drop penetration <0.1 otherwise you will get unrealistic UCS values
    df.drop(df[df['Penetration [mm/rot]'] < 0.1].index, inplace=True)
    # calculate UCS after Gehring 1995
    # p = a * sigmaD ^ -b (p=penetration, a, b=Funktionsparameter, simgD=UCS)
    # Funktionsparameter nach Ansatz Farmer: a = 729, b = 0.98
    df['UCS [MPa]'] = (729 / df['Penetration [mm/rot]']) ** 1/0.98
    # df['UCS [MPa]'].hist(bins=15, range=(50,350))
    ###########################################################################
    # prepare data for GAN
    df = df[1000:] # drop rows up to TM 1000 --> TBM personell learning phase
    # very narrow data-point-spacing --> keep only every 10th row
    df = df.iloc[::10]
    interval = interval * 10 # because only every 10th row kept
    look_back = 409.6 # results in a vector length of 8192
    look_back = int(look_back / interval) # 409.6/0.05 = 8192

    # scale data for training purposes
    def scale(feature, look_back):
        data = df[feature].values
        data = data.reshape(-1, 1)
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        data = data.squeeze()
        
        train_X = []
        # sliding window with the size 'look_back' to generate vectors of the 
        # desired size for the training data
        for i in np.arange(len(df), step=1):
             if i > look_back:
                train_X.append(data[i-look_back :i])
                
        train_X = np.asarray(train_X)

        return train_X, scaler
    
    data_pene, scaler_pene = scale('Penetration [mm/rot]', look_back)
    data_adv_force, scaler_adv_force = scale('Total advance force [kN]', look_back)
    data_torque, scaler_torque = scale('Torque cutterhead [MNm]', look_back)
    data_ucs, scaler_ucs = scale('UCS [MPa]', look_back)
        
    train_data = np.stack((data_pene, data_adv_force, data_torque, data_ucs), axis=1)
    print(train_data.shape)
    
    # save training data and scalers for later rescaling
    np.save(fr'00_data\03_train\GKI\GKI_train_X_pene_adv_force_torque_ucs_{look_back}.npy', train_data)
    dump(scaler_pene, open(fr'00_data\03_train\GKI\GKI_{machine}_scaler_pene_{look_back}.pkl', 'wb'))
    dump(scaler_adv_force, open(fr'00_data\03_train\GKI\GKI_{machine}_scaler_adv_force_{look_back}.pkl', 'wb'))
    dump(scaler_torque, open(fr'00_data\03_train\GKI\GKI_{machine}_scaler_torque_{look_back}.pkl', 'wb'))
    dump(scaler_ucs, open(fr'00_data\03_train\GKI\GKI_{machine}_scaler_ucs_{look_back}.pkl', 'wb'))
    
    return df, scaler_pene, scaler_adv_force, scaler_torque


df_GKI, GKI_scaler_pene, GKI_scaler_adv_force, GKI_scaler_torque = preprocessor(df, machine)
