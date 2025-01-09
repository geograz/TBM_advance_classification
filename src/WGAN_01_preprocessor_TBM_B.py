# -*- coding: utf-8 -*-
"""
Challenges and Opportunities of Data Driven Advance Classification for Hard
Rock TBM excavations

---- script to paper
DOI: XXXXXXX

Preprocessing functions for the TBM_B dataset

@author: Paul Unterlass
Created on Thu Jun 13 2024
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from WGAN_00_utilities import utilities
from pickle import dump

utils = utilities()

# =============================================================================
# combine single raw data files
# =============================================================================

def concat():
    # import raw data and concat individual excel files to big parquet file/csv
    machine = 'S980'
    folder = fr'00_data\01_raw\04_TBM_B\{machine}_TBM_Data'
    
    drop_standstills = True  # change to true if standstills should be dropped
    check_for_miss_vals = False  # generates plot of missing values in dataset
    
    df = utils.concat_tables_TBM_B(folder, drop_standstills=drop_standstills,
                                check_for_miss_vals=check_for_miss_vals)
    
    print('\ndataset start - stop:', df['Tunnel Distance [m]'].min(), '-', df['Tunnel Distance [m]'].max())
    print('dataset end of Time:', df['Timestamp'].max())
    
    
    if drop_standstills is False:
        df.to_parquet(fr'00_data\02_combined\TBM_B_TBMdata_wStandstills_{machine}.gzip', index=False)
    else:
        df.to_parquet(fr'00_data\02_combined\TBM_B_TBMdata_{machine}.gzip', index=False)
    
    return df, machine

df, machine = concat()

# =============================================================================
# preprocessing pipeline to create training data
# =============================================================================

# import pandas as pd
# df = pd.read_parquet(fr'00_data\02_combined\TBM_B_TBMdata_{machine}.gzip')

def preprocessor(df, machine):
    print('\n# datapoints', df['Tunnel Distance [m]'].count())
    
    df = df.loc[:,['Tunnel Distance [m]',
                   'CH Penetration [mm/rot]',
                   'Thrust Force [kN]',
                   'CH Torque [MNm]', 
                  ]]
    
    df = df.rename(columns={'CH Penetration [mm/rot]': 'Penetration [mm/rot]',
                            'Thrust Force [kN]': 'Total advance force [kN]',
                            'CH Torque [MNm]': 'Torque cutterhead [MNm]'})
    
    df = df[(df['Tunnel Distance [m]'] >=0 | (df['Tunnel Distance [m]'].isnull()))]
    df_median = df.groupby('Tunnel Distance [m]', as_index = False).median()
    print('\n# datapoints after grouping by Tunnel Distance', df_median['Tunnel Distance [m]'].count())

    df = df.sort_values(by=['Tunnel Distance [m]'])
    df.index = np.arange(len(df))
    ###########################################################################
    # linear interpolation
    # difference in Tunnel Distance between single data points
    interval = df_median['Tunnel Distance [m]'].diff().median()
    interval = round(interval, 3)
    df_equal = utils.equally_spaced_df(df_median, 'Tunnel Distance [m]', interval)
    print('# datapoints after linear interp.',
          df_equal['Tunnel Distance [m]'].count())
    df = df_equal
    ###########################################################################
    # drop penetration <0.1 otherwise you will get unrealistic UCS values
    df.drop(df[df['Penetration [mm/rot]'] < 0.1].index, inplace=True)
    # calculate UCS after Gehring 1995
    # p = a * sigmaD ^ -b (p=penetration, a, b=Funktionsparameter, simgD=UCS)
    # Funktionsparameter nach Ansatz Farmer: a = 729, b = 0.98
    df['UCS [MPa]'] = (729 / df['Penetration [mm/rot]']) ** 1/0.98
    ###########################################################################
    df['Tunnel Distance [m] round'] = np.round(df['Tunnel Distance [m]'], 3)
    df.set_index('Tunnel Distance [m] round', inplace=True, drop=False)
    df = df.drop(['Tunnel Distance [m] round', ], axis=1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.isna().sum()
    df = df.dropna()
    ###########################################################################
    # prepare data for GAN
    df = df[1000:] # drop rows up to TM 1000
    df = df.iloc[::10] # keeps only every 5th row
    interval = interval * 10 # because only every 5th row kept
    look_back = 204.8 # results in a vector length of 8192
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
    np.save(fr'00_data\03_train\TBM_B\TBM_B_train_X_pene_adv_force_torque_ucs_{look_back}_{machine}.npy', train_data)
    dump(scaler_pene, open(fr'00_data\03_train\TBM_B\TBM_B_{machine}_scaler_pene_{look_back}.pkl', 'wb'))
    dump(scaler_adv_force, open(fr'00_data\03_train\TBM_B\TBM_B_{machine}_scaler_adv_force_{look_back}.pkl', 'wb'))
    dump(scaler_torque, open(fr'00_data\03_train\TBM_B\TBM_B_{machine}_scaler_torque_{look_back}.pkl', 'wb'))
    dump(scaler_ucs, open(fr'00_data\03_train\TBM_B\TBM_B_{machine}_scaler_ucs_{look_back}.pkl', 'wb'))
    
    return df, scaler_pene, scaler_adv_force, scaler_torque, scaler_ucs


df_TBM_B, TBM_B_scaler_pene, TBM_B_scaler_adv_force, TBM_B_scaler_torque, TBM_B_scaler_ucs = preprocessor(df, machine)
