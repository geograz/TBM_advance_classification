# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 21:09:10 2023

@author: GEr
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DATA_XX_library import utilities


######################################
# fixed values and variables
######################################

SAMPLE = 'Ulriken'  # 'BBT' 'Ulriken' 'Follo'

######################################
# data import
######################################

# instantiations
utils = utilities(SAMPLE)

# load one dataset
fname = utils.param_dict['filename']
df = pd.read_csv(f'../data/{fname}_mod.csv')

######################################
# preprocessing
######################################

np.random.seed(123)  # fix random seed for reproducibility

# remove standstills
df_advance = df[(df['Penetration [mm/rot]'] > 0) &
                (df['Total advance force [kN]'] > 0) &
                (df['Torque cutterhead [MNm]'] > 0) &
                (df['rotations [rpm]'] > 0)]

# make dataframe with stroke wise values
df_strokes = []
for stroke in df_advance.groupby('Stroke number [-]'):
    df_stroke_temp = pd.DataFrame({
        'Stroke number [-]': [stroke[0]],
        'tunnellength stroke start [m]': [stroke[1]['tunnellength [m]'].min()],
        'tunnellength stroke end [m]': [stroke[1]['tunnellength [m]'].max()],
        'time stroke start [m]': [stroke[1]['Timestamp'].min()],
        'time stroke end [m]': [stroke[1]['Timestamp'].max()],
        'Penetration [mm/rot] mean': [stroke[1]['Penetration [mm/rot]'].mean()],
        'Total advance force [kN] mean': [stroke[1]['Total advance force [kN]'].mean()],
        'Torque cutterhead [MNm] mean': [stroke[1]['Torque cutterhead [MNm]'].mean()],
        'rotations [rpm] mean': [stroke[1]['rotations [rpm]'].mean()],
        'Penetration [mm/rot] median': [stroke[1]['Penetration [mm/rot]'].median()],
        'Total advance force [kN] median': [stroke[1]['Total advance force [kN]'].median()],
        'Torque cutterhead [MNm] median': [stroke[1]['Torque cutterhead [MNm]'].median()],
        'rotations [rpm] median': [stroke[1]['rotations [rpm]'].median()]
        })
    df_stroke_temp['tunnellength stroke middle [m]'] = df_stroke_temp[['tunnellength stroke start [m]',
                                                                       'tunnellength stroke end [m]']].values.mean()
    df_strokes.append(df_stroke_temp)

df_strokes = pd.concat(df_strokes)

# compute theoretical cutterhead torque & torque ratio for both dfs
df['theo. torque [kNm]'], df['torque ratio [-]'] = utils.torque_ratio(
    df['Total advance force [kN]'], df['Penetration [mm/rot]'],
    df['Torque cutterhead [MNm]']*1000)

df_strokes['theo. torque [kNm] mean'], df_strokes['torque ratio [-] mean'] = utils.torque_ratio(
    df_strokes['Total advance force [kN] mean'],
    df_strokes['Penetration [mm/rot] mean'],
    df_strokes['Torque cutterhead [MNm] mean']*1000)
df_strokes['theo. torque [kNm] median'], df_strokes['torque ratio [-] median'] = utils.torque_ratio(
    df_strokes['Total advance force [kN] median'],
    df_strokes['Penetration [mm/rot] median'],
    df_strokes['Torque cutterhead [MNm] median']*1000)

df_strokes.to_excel(f'../data/{fname}_mod.xlsx', index=False)

######################################
# plotting
######################################


fig = plt.figure(figsize=(12, 8))

x1, x2 = 'tunnellength [m]', 'tunnellength stroke middle [m]'
length = 30

for i, parameter in enumerate(['Penetration [mm/rot]',
                               'Total advance force [kN]',
                               'Torque cutterhead [MNm]',
                               'torque ratio [-]']):
    df = df[df[x1] < length]
    df_strokes = df_strokes[df_strokes[x2] < length]

    ax = fig.add_subplot(4, 1, i+1)
    ax.plot(df[x1], df[parameter], color='grey', label='TBM data', zorder=5)
    ax.scatter(df_strokes[x2], df_strokes[parameter+' mean'],
               label='stroke mean', s=60, color='white', marker='o',
               edgecolor='black', zorder=10)
    ax.scatter(df_strokes[x2], df_strokes[parameter+' median'],
               label='stroke median', s=60, color='white', marker='v',
               edgecolor='black', zorder=10)

    ax_lims = ax.get_ylim()
    ax.vlines(x=df_strokes['tunnellength stroke start [m]'], ymin=ax_lims[0],
              ymax=ax_lims[1], color='black', label='stroke boundary',
              zorder=20)
    ax.set_ylabel(parameter)
    ax.set_xlim(left=0, right=length)
    ax.grid(alpha=0.5)

ax.legend(bbox_to_anchor=(1, 0.5))
ax.set_xlabel(x1)

plt.tight_layout()
plt.savefig(f"../figures/{utils.param_dict['plotname']}.png")
plt.close()
