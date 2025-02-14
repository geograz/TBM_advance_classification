# -*- coding: utf-8 -*-
"""
Challenges and Opportunities of Data Driven Advance Classification for Hard
Rock TBM excavations

---- script to paper
DOI: XXXXXXX

Code processes "raw" TBM operational data according to the 4 major steps
described in the paper.

@author: Dr. Georg Erharter, Theresa Maier
"""

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

# load the raw TBM dataset from a zip file
fname = utils.param_dict['filename']
df = pd.read_csv(f'../data/{fname}_1_synthetic_realistic.zip')

######################################
# Basic data cleaning
######################################

np.random.seed(123)  # fix random seed for reproducibility

# remove standstills
df_advance = df[(df['Penetration [mm/rot]'] > 0) |
                (df['Total advance force [kN]'] > 0) |
                (df['Torque cutterhead [MNm]'] > 0) |
                (df['rotations [rpm]'] > 0)]

# average datapoints with same tunnellength
df_advance = df_advance.drop(
    'Timestamp', axis=1).groupby('tunnellength [m]', as_index=False).mean()

######################################
# Spatial discretization
######################################

# make dataframe with stroke wise values
df_strokes = []
for stroke in df_advance.groupby('Stroke number [-]'):
    df_stroke_temp = pd.DataFrame({
        'Stroke number [-]': [stroke[0]],
        'tunnellength stroke start [m]': [stroke[1]['tunnellength [m]'].min()],
        'tunnellength stroke end [m]': [stroke[1]['tunnellength [m]'].max()],
        'Penetration [mm/rot] mean': [stroke[1]['Penetration [mm/rot]'].mean()],
        'Total advance force [kN] mean': [stroke[1]['Total advance force [kN]'].mean()],
        'Torque cutterhead [MNm] mean': [stroke[1]['Torque cutterhead [MNm]'].mean()],
        'rotations [rpm] mean': [stroke[1]['rotations [rpm]'].mean()],
        'Penetration [mm/rot] median': [stroke[1]['Penetration [mm/rot]'].median()],
        'Total advance force [kN] median': [stroke[1]['Total advance force [kN]'].median()],
        'Torque cutterhead [MNm] median': [stroke[1]['Torque cutterhead [MNm]'].median()],
        'rotations [rpm] median': [stroke[1]['rotations [rpm]'].median()]
        })
    df_stroke_temp['tunnellength stroke middle [m]'] = df_stroke_temp[
        ['tunnellength stroke start [m]',
         'tunnellength stroke end [m]']].values.mean()
    df_strokes.append(df_stroke_temp)
df_strokes = pd.concat(df_strokes)

######################################
# Parameter computation
######################################

# compute theoretical cutterhead torque & torque ratio for all dfs
df_advance['theo. torque [kNm]'], df_advance['torque ratio [-]'] = utils.torque_ratio(
    df_advance['Total advance force [kN]'], df_advance['Penetration [mm/rot]'],
    df_advance['Torque cutterhead [MNm]']*1000)

df_strokes['theo. torque [kNm] mean'], df_strokes['torque ratio [-] mean'] = utils.torque_ratio(
    df_strokes['Total advance force [kN] mean'],
    df_strokes['Penetration [mm/rot] mean'],
    df_strokes['Torque cutterhead [MNm] mean']*1000)
df_strokes['theo. torque [kNm] median'], df_strokes['torque ratio [-] median'] = utils.torque_ratio(
    df_strokes['Total advance force [kN] median'],
    df_strokes['Penetration [mm/rot] median'],
    df_strokes['Torque cutterhead [MNm] median']*1000)

######################################
# Threshold definition and advance classification
######################################

df_strokes['advance class mean'] = np.where(
    (df_strokes['torque ratio [-] mean'] > utils.param_dict['torque ratio bounds'][0])&
    (df_strokes['torque ratio [-] mean'] < utils.param_dict['torque ratio bounds'][1]),
    0, 1)
df_strokes['advance class median'] = np.where(
    (df_strokes['torque ratio [-] median'] > utils.param_dict['torque ratio bounds'][0])&
    (df_strokes['torque ratio [-] median'] < utils.param_dict['torque ratio bounds'][1]),
    0, 1)

######################################
# Save files
######################################

df_advance.to_excel(f'../data/{fname}_2_synthetic_advance.xlsx', index=False)
df_strokes.to_excel(f'../data/{fname}_2_synthetic_strokes.xlsx', index=False)

######################################
# Print number of strokes achieved with arithmetic mean and median:
#   - Strokes regular advance
#   - Strokes exceptional advance
######################################

n_strokes = len(df_strokes)
n_regular_advance_mean = df_strokes[df_strokes['advance class mean'] == 0].shape[0]
n_exceptional_advance_mean = df_strokes[df_strokes['advance class mean'] == 1].shape[0]
ratio_mean_regular = round(n_regular_advance_mean/n_strokes*100, 1)
ratio_mean_exceptional = round(n_exceptional_advance_mean/n_strokes*100, 1)

n_regular_advance_median = df_strokes[df_strokes['advance class median'] == 0].shape[0]
n_exceptional_advance_median = df_strokes[df_strokes['advance class median'] == 1].shape[0]
ratio_median_regular = round(n_regular_advance_median/n_strokes*100, 1)
ratio_median_exceptional = round(n_exceptional_advance_median/n_strokes*100, 1)

print(f'{SAMPLE} Regular advance:')
print(f"n strokes based on arithmetic mean: {n_regular_advance_mean}")
print(f"n strokes based on median: {n_regular_advance_median}\n")
print(f'{SAMPLE} Exceptional advance:')
print(f"n strokes based on arithmetic mean: {n_exceptional_advance_mean}")
print(f"n strokes based on median: {n_exceptional_advance_median}\n")
print(f'ratio regular/exceptional based on arithmetic mean [%]: {ratio_mean_regular}/{ratio_mean_exceptional}')
print(f'ratio regular/exceptional based on median [%]: {ratio_median_regular}/{ratio_median_exceptional}')
