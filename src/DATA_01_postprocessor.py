# -*- coding: utf-8 -*-
"""
Challenges and Opportunities of Data Driven Advance Classification for Hard
Rock TBM excavations

---- script to paper
DOI: XXXXXXX

Code takes synthetic TBM operational data that was created by GANs and post-
processes it to make it look more realistic.

@author: Dr. Georg Erharter
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DATA_XX_library import utilities


######################################
# fixed values and variables
######################################

SAMPLE = 'TBM_A'  # 'TBM_A' 'TBM_B' 'TBM_C'

######################################
# data import
######################################

# instantiations
utils = utilities(SAMPLE)

# load one dataset
fname = utils.param_dict['filename']
df = pd.read_parquet(f'../data/{fname}_0_synthetic_raw')

######################################
# preprocessing
######################################

np.random.seed(123)  # fix random seed for reproducibility

# backcalculate tunnel distance from penetration and cutterhead rotations
df['rotations [rpm]'] = utils.param_dict['cutterhead rotations']
increment = df['Penetration [mm/rot]'] / df['rotations [rpm]'] / 1000
df['tunnellength [m]'] = np.round(np.cumsum(increment), 2)

# generate stroke numbers
df['Stroke number [-]'] = 0
n_strokes = int(df['tunnellength [m]'].max()/utils.param_dict['stroke length'])
for stroke in np.arange(n_strokes+1):
    idx = df[(df['tunnellength [m]'] >= stroke*utils.param_dict['stroke length']) &
             (df['tunnellength [m]'] < (stroke+1)*utils.param_dict['stroke length'])].index
    df.loc[idx, 'Stroke number [-]'] = np.full(len(idx), stroke)


# insert stillstand after advance
strokes = []
for stroke in np.arange(df['Stroke number [-]'].max()):
    idx = df[df['Stroke number [-]'] == stroke].index
    strokes.append(df.loc[idx])
    # make dataframe for standstill data
    df_stillstand_temp = pd.DataFrame(columns=df.columns)
    length_standstill = np.random.uniform(0.7, 2)  # duration of standstill [h]
    n_dp_standstill = int(length_standstill*60*60/utils.param_dict['frequency'])
    df_stillstand_temp['Penetration [mm/rot]'] = np.zeros(n_dp_standstill)
    df_stillstand_temp['Total advance force [kN]'] = np.zeros(n_dp_standstill)
    df_stillstand_temp['Torque cutterhead [MNm]'] = np.zeros(n_dp_standstill)
    df_stillstand_temp['rotations [rpm]'] = np.zeros(n_dp_standstill)
    df_stillstand_temp['tunnellength [m]'] = np.full(
        n_dp_standstill, df.loc[idx]['tunnellength [m]'].max())
    df_stillstand_temp['Stroke number [-]'] = np.full(
        n_dp_standstill, df.loc[idx]['Stroke number [-]'].max())
    strokes.append(df_stillstand_temp)
df = pd.concat(strokes)
df.index = np.arange(len(df))

# generate timestamps
df['Timestamp'] = pd.date_range(start='1/1/2024', periods=len(df),
                                freq=f"{utils.param_dict['frequency']}s")

# reorder dataframe columns
df = df[['Timestamp', 'tunnellength [m]', 'Stroke number [-]',
         'Penetration [mm/rot]', 'Total advance force [kN]',
         'Torque cutterhead [MNm]', 'rotations [rpm]']]
# trim to 1000 meters max
df = df[df['tunnellength [m]'] <= 1000]

# save dataframe to zipped .csv file
compression_opts = dict(method='zip', archive_name=f'{fname}_mod.csv')
df.to_csv(f'../data/{fname}_1_synthetic_realistic.zip', index=False,
          compression=compression_opts)
