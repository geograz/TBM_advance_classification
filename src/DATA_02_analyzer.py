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

SAMPLE = 'TBM_A'  # 'TBM_A' 'TBM_B' 'TBM_C'

######################################
# data import
######################################

# instantiations
utils = utilities(SAMPLE)

# load the raw TBM dataset from a zip file
fname = utils.param_dict['filename']
df = pd.read_csv(f'../data/{fname}_mod.zip')

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
df['theo. torque [kNm]'], df['torque ratio [-]'] = utils.torque_ratio(
    df['Total advance force [kN]'], df['Penetration [mm/rot]'],
    df['Torque cutterhead [MNm]']*1000)

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

df_strokes.to_excel(f'../data/{fname}_mod.xlsx', index=False)

######################################
# Threshold definition and advance classification
######################################

# TODO

######################################
# plotting
######################################

# TODO make dedicated script for plotting

# scaling of advance force
df['Total advance force [MN]'] = df['Total advance force [kN]'] / 1000
df_advance['Total advance force [MN]'] = df_advance['Total advance force [kN]'] / 1000
df_strokes['Total advance force [MN] mean'] = df_strokes['Total advance force [kN] mean'] / 1000
df_strokes['Total advance force [MN] median'] = df_strokes['Total advance force [kN] median'] / 1000

for start in np.arange(1000, step=100):
    fig = plt.figure(figsize=(7.48031, 3.93701))

    x1, x2 = 'tunnellength [m]', 'tunnellength stroke middle [m]'
    length = 20
    fontsize = 6

    for i, parameter in enumerate(['Penetration [mm/rot]',
                                   'Total advance force [MN]',
                                   'Torque cutterhead [MNm]',
                                   'torque ratio [-]']):
        df_ = df[(df[x1] >= start) & (df[x1] < start+length)]
        df_advance_ = df_advance[(df_advance[x1] >= start) &
                                 (df_advance[x1] < start+length)]
        df_strokes_ = df_strokes[(df_strokes[x2] >= start) &
                                 (df_strokes[x2] < start+length)]

        ax = fig.add_subplot(4, 1, i+1)
        ax.scatter(df_[x1], df_[parameter], color='grey', label='raw TBM data',
                   zorder=5, alpha=0.3, s=4)
        ax.plot(df_advance_[x1], df_advance_[parameter], color='black',
                label='TBM data cleaned', lw=1, zorder=7)
        ax.scatter(df_strokes_[x2], df_strokes_[parameter+' mean'],
                   label='stroke mean', s=30, color='white', marker='o',
                   edgecolor='black', zorder=10)
        ax.scatter(df_strokes_[x2], df_strokes_[parameter+' median'],
                   label='stroke median', s=30, color='white', marker='v',
                   edgecolor='black', zorder=10)
        ax.set_xlim(left=start, right=start+length)
        ax_lims = ax.get_ylim()
        if parameter == 'torque ratio [-]' and df_[parameter].max() > 1.5:
            ax_lims = (ax_lims[0], 1.5)
            ax.set_ylim(top=1.6)
        ax.vlines(x=df_strokes_['tunnellength stroke start [m]'],
                  ymin=ax_lims[0], ymax=ax_lims[1], color='black',
                  label='stroke boundary', zorder=20)
        ax.set_ylabel(parameter.replace(' [', '\n['), fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)

        if i == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=5,
                      fontsize=fontsize)

        ax.grid(alpha=0.5)

    ax.set_xlabel(x1, fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(f"../figures/{utils.param_dict['plotname']}_{start}.png",
                dpi=400)
    plt.close()
