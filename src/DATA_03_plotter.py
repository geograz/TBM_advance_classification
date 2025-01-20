# -*- coding: utf-8 -*-
"""
Challenges and Opportunities of Data Driven Advance Classification for Hard
Rock TBM excavations

---- script to paper
DOI: XXXXXXX

Code creates plots of the synthetic data for the paper.

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

# load the raw TBM dataset from a zip file
fname = utils.param_dict['filename']
df = pd.read_csv(f'../data/{fname}_1_synthetic_realistic.zip')
df_advance = pd.read_excel(f'../data/{fname}_2_synthetic_advance.xlsx')
df_strokes = pd.read_excel(f'../data/{fname}_2_synthetic_strokes.xlsx')

######################################
# compute parameters for plotting
######################################


p_regular_mean = np.unique(
    df_strokes['advance class mean'],
    return_counts=True)[1][0] / len(df_strokes) * 100
p_exceptional_mean = np.unique(
    df_strokes['advance class mean'],
    return_counts=True)[1][1] / len(df_strokes) * 100
p_regular_median = np.unique(
    df_strokes['advance class median'],
    return_counts=True)[1][0] / len(df_strokes) * 100
p_exceptional_median = np.unique(
    df_strokes['advance class median'],
    return_counts=True)[1][1] / len(df_strokes) * 100
print(SAMPLE)
print(f'mean - regular: {round(p_regular_mean, 1)}, exceptional: {round(p_exceptional_mean, 1)}')
print(f'median - regular: {round(p_regular_median, 1)}, exceptional: {round(p_exceptional_median, 1)}')

df['theo. torque [kNm]'], df['torque ratio [-]'] = utils.torque_ratio(
    df['Total advance force [kN]'], df['Penetration [mm/rot]'],
    df['Torque cutterhead [MNm]']*1000)

# scaling of advance force
df['Total advance force [MN]'] = df['Total advance force [kN]'] / 1000
df_advance['Total advance force [MN]'] = df_advance['Total advance force [kN]'] / 1000
df_strokes['Total advance force [MN] mean'] = df_strokes['Total advance force [kN] mean'] / 1000
df_strokes['Total advance force [MN] median'] = df_strokes['Total advance force [kN] median'] / 1000

######################################
# plotting
######################################

for start in np.arange(1000, step=100):
    fig = plt.figure(figsize=(7.48031, 4.72441))

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
                   zorder=5, alpha=0.2, s=1)
        ax.plot(df_advance_[x1], df_advance_[parameter], color='black',
                label='TBM data cleaned', lw=0.5, zorder=7)
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

        # torque ratio specific plot adjustments
        if parameter == 'torque ratio [-]':
            # plot upper and lower boundaries of torque ratio
            ax.hlines(y=utils.param_dict['torque ratio bounds'], color='black',
                      ls='--', xmin=start, xmax=start+length, lw=1, zorder=30)
            for i in df_strokes_.index:  # stroke specific things
                # color strokes according to advance classification
                if df_strokes_.loc[i]['advance class mean'] == 0 and df_strokes_.loc[i]['advance class median'] == 0:
                    ax.axvspan(
                        xmin=df_strokes_.loc[i]['tunnellength stroke start [m]'],
                        xmax=df_strokes_.loc[i]['tunnellength stroke end [m]'],
                        color='green', alpha=0.8, zorder=0)
                elif df_strokes_.loc[i]['advance class mean'] == 1 and df_strokes_.loc[i]['advance class median'] == 1:
                    ax.axvspan(
                        xmin=df_strokes_.loc[i]['tunnellength stroke start [m]'],
                        xmax=df_strokes_.loc[i]['tunnellength stroke end [m]'],
                        color='red', alpha=0.8, zorder=0)
                else:
                    ax.axvspan(
                        xmin=df_strokes_.loc[i]['tunnellength stroke start [m]'],
                        xmax=df_strokes_.loc[i]['tunnellength stroke end [m]'],
                        color='orange', alpha=0.8, zorder=0)
                # add strokes numbers
                ax.text(x=df_strokes_.loc[i]['tunnellength stroke middle [m]'],
                        y=ax.get_ylim()[0]+0.05,
                        s='stroke\n'+str(int(df_strokes_.loc[i]['Stroke number [-]'])),
                        ha='center', va='bottom', fontsize=fontsize)

        # plot stroke boundaries
        boundaries = list(df_strokes_['tunnellength stroke start [m]']) + \
            [df_strokes_['tunnellength stroke end [m]'].iloc[-1]]
        ax.vlines(x=boundaries, ymin=ax_lims[0], ymax=ax_lims[1],
                  color='black', label='stroke boundary', lw=1, zorder=20)
        ax.set_ylabel(parameter.replace(' [', '\n['), fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)

        if i == 0:  # make a legend above all plots
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=5,
                      fontsize=fontsize)

        if i < 3:  # hide x tick labels for all but the last axis
            xax = ax.axes.get_xaxis()
            xax = xax.set_visible(False)

        ax.grid(alpha=0.5, axis='y')

    ax.set_xlabel(x1, fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(f"../figures/{utils.param_dict['filename']}_{start}.png",
                dpi=400)
    plt.savefig(f"../figures/{utils.param_dict['filename']}_{start}.svg")
    plt.savefig(f"../figures/{utils.param_dict['filename']}_{start}.pdf")
    plt.close()
