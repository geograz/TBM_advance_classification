# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 21:09:10 2023

@author: GEr
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


########################
# function definition
########################

def torque_ratio(advance_force: pd.Series,  # cutterhead advance force [kN]
                 n_cutters: int,  # number of cutters on cutterhead
                 cutter_radius: float,  # cutter radius [mm]
                 penetration: pd.Series,  # penetration [mm/rev]
                 cutterhead_diameter: float,  # cutterhead diameter [m]
                 M0: float,  # empty torque [kNm]
                 torque: pd.Series  # real torque [kNm]
                 ) -> pd.Series:
    '''computation of the torque ratio according to the Austrian
    contractual standard: ÖNORM B2203-2 (2023)'''
    F_n = advance_force / n_cutters
    alpha = np.arccos((cutter_radius - penetration)/cutter_radius)
    F_tang = np.tan(alpha/2)*F_n
    theo_torque = 0.3 * F_tang * n_cutters * cutterhead_diameter + M0
    torque_ratio = torque / theo_torque
    return theo_torque, torque_ratio


######################################
# fixed values and variables
######################################

SAMPLE = 'BBT'
STROKE_LENGTH = 1.8  # [m] length of one stroke of the TBM

######################################
# data import and preprocessing
######################################


match SAMPLE:  # noqa
    case 'Ulriken':
        param_dict = {'filename': 'UT_synth_pene_adv_force_torque_1024.npy',
                      'plotname': 'sample_Ulriken.png',
                      'idx': 230,  # data index to load
                      'dp spacing': 0.03,  # [m] data point spacing
                      'n cutters': 50,
                      'cutter radius': 241.3,  # [mm]
                      'cutterhead diameter': 8,  # [m]
                      'M0': 400  # [kNm]
                      }
    case 'BBT':
        param_dict = {'filename': 'BBT_synth_pene_adv_force_torque_1024.npy',
                      'plotname': 'sample_BBT.png',
                      'idx': 4,  # data index to load
                      'dp spacing': 0.05,  # [m] data point spacing
                      'n cutters': 50,
                      'cutter radius': 216,  # [mm]
                      'cutterhead diameter': 6,  # [m]
                      'M0': 300  # [kNm]
                      }


# load one dataset from numpy array
data = np.load(param_dict['filename'])
sample_penetration = data[:, 0, :][param_dict['idx']].flatten()
sample_tot_adv_force = data[:, 1, :][param_dict['idx']].flatten()
sample_torque = data[:, 2, :][param_dict['idx']].flatten()
n_dp = len(sample_penetration)


# create pandas dataframe with TBM operational data
df = pd.DataFrame({'tunnellength [m]': np.arange(n_dp)*param_dict['dp spacing'],
                   'penetration [mm/rev]': sample_penetration,
                   'total advance force [kN]': sample_tot_adv_force,
                   'torque [kNm]': sample_torque*1000})
# generate stroke numbers
df['Stroke number [-]'] = (df['tunnellength [m]'] / STROKE_LENGTH).astype(int)
# get stroke wise averages
stroke_starts = df.groupby(['Stroke number [-]'],
                           as_index=False).min()['tunnellength [m]'].values
df_strokes = df.groupby(
    ['Stroke number [-]'], as_index=False).mean(numeric_only=True)
df_strokes.drop('tunnellength [m]', axis=1, inplace=True)
df_strokes['tunnellength stroke start [m]'] = stroke_starts
df_strokes['tunnellength stroke middle [m]'] = stroke_starts + STROKE_LENGTH/2

# compute theoretical cutterhead torque & torque ratio for both dfs
df['theo. cutterhead torque [kNm]'], df['torque ratio [-]'] = torque_ratio(
    df['total advance force [kN]'], param_dict['n cutters'],
    param_dict['cutter radius'], df['penetration [mm/rev]'],
    param_dict['cutterhead diameter'], param_dict['M0'], df['torque [kNm]'])
df_strokes['theo. cutterhead torque [kNm]'], df_strokes['torque ratio [-]'] = torque_ratio(
    df_strokes['total advance force [kN]'], param_dict['n cutters'],
    param_dict['cutter radius'], df_strokes['penetration [mm/rev]'],
    param_dict['cutterhead diameter'], param_dict['M0'], df_strokes['torque [kNm]'])

print(np.mean(df['torque ratio [-]']))
######################################
# plotting
######################################


fig = plt.figure(figsize=(12, 8))

x1, x2 = 'tunnellength [m]', 'tunnellength stroke middle [m]'

for i, parameter in enumerate(['penetration [mm/rev]',
                               'total advance force [kN]', 'torque [kNm]',
                               'torque ratio [-]']):
    ax = fig.add_subplot(4, 1, i+1)
    ax.plot(df[x1], df[parameter], color='grey', label='TBM data', zorder=5)
    ax.scatter(df_strokes[x2], df_strokes[parameter],
               label='stroke wise average', s=60, color='white',
               edgecolor='black', zorder=10)
    ax_lims = ax.get_ylim()
    ax.vlines(x=df_strokes['tunnellength stroke start [m]'], ymin=ax_lims[0],
              ymax=ax_lims[1], color='black', label='stroke boundary',
              zorder=20)
    ax.set_ylabel(parameter)
    ax.set_xlim(left=0, right=18*STROKE_LENGTH)
    # ax.grid(alpha=0.5)

ax.legend(bbox_to_anchor=(1, 0.5))
ax.set_xlabel(x1)

plt.tight_layout()
plt.savefig(param_dict['plotname'])
# plt.close()
