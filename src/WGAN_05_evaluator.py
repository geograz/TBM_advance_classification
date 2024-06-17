# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 2024

@author: Paul Unterlass


"""

from table_evaluator import TableEvaluator
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

tunnel = 'TBM_B' # TBM_A
FEATURE_NAME1 = 'pene'
FEATURE_NAME2 = 'adv_force'
FEATURE_NAME3 = 'torque'
FEATURE_NAME4 = 'ucs'
features = [FEATURE_NAME1, FEATURE_NAME2, FEATURE_NAME3, FEATURE_NAME4]

# filepath = r'C:/02_Research/01_Unterlass/01_GANs/03_GANs/06_WGAN_oenorm/01_results/TBM_A_2024-06-13_15-07-47_bs32_gen426_pene_adv_force_torque_ucs_4096'
# filepath = r'C:/02_Research/01_Unterlass/01_GANs/03_GANs/06_WGAN_oenorm/01_results/TBM_C_2024-06-13_15-27-09_bs32_gen301_pene_adv_force_torque_ucs_4096'
filepath = r'C:/02_Research/01_Unterlass/01_GANs/03_GANs/06_WGAN_oenorm/01_results/TBM_B_2024-06-13_14-07-27_bs32_gen226__pene_adv_force_torque_ucs_4096'

look_back = 4096 #1024
N_PASSES = 250
run = 1
machine = 'S980'

# =============================================================================
# load data
# =============================================================================

if machine is not None:
    real_data = np.load(fr'00_data\03_train\{tunnel}\{tunnel}_train_X_{FEATURE_NAME1}_{FEATURE_NAME2}_{FEATURE_NAME3}_{FEATURE_NAME4}_{look_back}_{machine}.npy')
else:
    real_data = np.load(fr'00_data\03_train\{tunnel}\{tunnel}_train_X_{FEATURE_NAME1}_{FEATURE_NAME2}_{FEATURE_NAME3}_{FEATURE_NAME4}_{look_back}.npy')

SEQ_LENGTH = real_data.shape[2]
if machine is not None:    
    synth_data = np.load(fr'{filepath}/03_generated_data/{tunnel}_length{SEQ_LENGTH}_n{N_PASSES}_{FEATURE_NAME1}_{FEATURE_NAME2}_{FEATURE_NAME3}_{FEATURE_NAME4}_{run}_{machine}.npy')
else:
    synth_data = np.load(fr'{filepath}/03_generated_data/{tunnel}_length{SEQ_LENGTH}_n{N_PASSES}_{FEATURE_NAME1}_{FEATURE_NAME2}_{FEATURE_NAME3}_{FEATURE_NAME4}_{run}.npy')

synth_data = synth_data[:len(real_data)]
real_data = real_data[:len(synth_data)]

real_data_pene = real_data[:, 0, :]
real_data_adv_force = real_data[:, 1, :]
real_data_torque = real_data[:, 2, :]
real_data_ucs = real_data[:, 3, :]

synth_data_pene = synth_data[:, 0, :]
synth_data_adv_force = synth_data[:, 1, :]
synth_data_torque = synth_data[:, 2, :]
synth_data_ucs = synth_data[:, 3, :]

# DEMAND FOR CONFORMITY
# =============================================================================
# PCA and tSNE
# =============================================================================

sample_size = 4096
num_samples = 2
N_ITER = 700
PERPLEXITY = 50

def PCA_tSNE(real_data, synth_data):
    num_samples = 2
    N_ITER = 700
    PERPLEXITY = 50

    idx = np.random.permutation(len(synth_data))[
        :num_samples]  # randomly permuted indizes
    real_sample = np.asarray(real_data)[idx]
    real_sample_pca = real_sample.reshape(-1, num_samples)
    synth_sample = np.asarray(synth_data)[idx]
    synth_sample_pca = synth_sample.reshape(-1, num_samples)

    # PCA
    n_components = 2  # 2D data
    pca = PCA(n_components=n_components)

    pca.fit(real_sample_pca)
    pca_real = pd.DataFrame(pca.transform(real_sample_pca))
    pca_synth = pd.DataFrame(pca.transform(synth_sample_pca))

    # TSNE
    all_data = np.concatenate((real_sample_pca, synth_sample_pca), axis=0)
    tsne = TSNE(n_components=n_components,
                n_iter=N_ITER, perplexity=PERPLEXITY)
    tsne_results = pd.DataFrame(tsne.fit_transform(all_data))

    return pca_real, pca_synth, tsne_results

pca_real_pene, pca_synth_pene, tsne_results_pene = PCA_tSNE(
    real_data_pene, synth_data_pene)
pca_real_adv_force, pca_synth_adv_force, tsne_results_adv_force = PCA_tSNE(
    real_data_adv_force, synth_data_adv_force)
pca_real_torque, pca_synth_torque, tsne_results_torque = PCA_tSNE(
    real_data_torque, synth_data_torque)
pca_real_ucs, pca_synth_ucs, tsne_results_ucs = PCA_tSNE(
    real_data_ucs, synth_data_ucs)

# =============================================================================
# visualize PCA and TSNE results
# =============================================================================
features_full = ['Penetration [mm/rot]', 'Total advance force [kN]',
                 'Torque cutterhead [MNm]', 'UCS [MPa]']

for i in range(len(features)):

    fig = plt.figure(constrained_layout=True, figsize=(20, 10), dpi=300)
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    # PCA scatter plot
    ax = fig.add_subplot(spec[0, 0])
    ax.set_title(f'PCA results {features_full[i]}',
                 fontsize=20,
                 pad=10)

    plt.scatter(eval(f'pca_real_{features[i]}.iloc[:,0].values'),
                eval(f'pca_real_{features[i]}.iloc[:,1].values'),
                c='black', alpha=0.5, label=f'real data {features_full[i]}',
                marker='x', s=70)
    plt.scatter(eval(f'pca_synth_{features[i]}.iloc[:,0].values'),
                eval(f'pca_synth_{features[i]}.iloc[:,1].values'),
                c='black', alpha=0.1, label=f'synthetic data {features_full[i]}',
                s=70)

    ax.tick_params(axis='both', which='major', labelsize='18')
    ax.legend(fontsize='18')

    # TSNE scatter plot
    ax2 = fig.add_subplot(spec[0, 1])
    ax2.set_title(f't-SNE results {features_full[i]}',
                  fontsize=20,
                  pad=10)

    plt.scatter(eval(f'tsne_results_{features[i]}.iloc[:{sample_size}, 0].values'),
                eval(
                    f'tsne_results_{features[i]}.iloc[:{sample_size}, 1].values'),
                c='black', alpha=0.5, label=f'real data {features_full[i]}',
                marker='x', s=70)
    plt.scatter(eval(f'tsne_results_{features[i]}.iloc[{sample_size}:, 0].values'),
                eval(
                    f'tsne_results_{features[i]}.iloc[{sample_size}:, 1].values'),
                c='black', alpha=0.1, label=f'synthetic data {features_full[i]}', s=70)
    ax2.tick_params(axis='both', which='major', labelsize='18')
    ax2.legend(fontsize='18')

    plt.savefig(
        f'{filepath}/02_PCA_tSNE/PCA, tSNE_{features[i]}_iter{N_ITER}_p{PERPLEXITY}_n_samples{num_samples}.png', dpi=300)

# DEMAND FOR ORIGINALITY
# =============================================================================
# similarity score
# =============================================================================

# check most similiar vectors
def inverse_scale_most_sim_vec(real, synth, features):

    real = pd.DataFrame()
    synth = pd.DataFrame()

    for i in range(len(features)):
        if machine is not None:
            file = open(fr'00_data\03_train\{tunnel}\{tunnel}_{machine}_scaler_{features[i]}_{look_back}.pkl', 'rb')
        else:
            file = open(fr'00_data\03_train\{tunnel}\{tunnel}_scaler_{features[i]}_{look_back}.pkl', 'rb')
        scaler = pickle.load(file)
        file.close()

        real_data = scaler.inverse_transform(eval(f'real_data_{features[i]}')) # eval to interpret string as variable
        synth_data = scaler.inverse_transform(eval(f'synth_data_{features[i]}'))

        with open(f'{filepath}/04_vector_distance/similar_vectors_{features[i]}', 'rb') as f:
            similar_vector = pickle.load(f)

        idx = np.random.permutation(len(synth_data))[1]

        real_data_sample = real_data[idx].squeeze() # take one random sample, get rid of 1st dimension
        synth_data_sample = synth_data[similar_vector[idx]].squeeze() # take most similar synth sample

        real[f'{features[i]}'] = real_data_sample
        synth[f'{features[i]}'] = synth_data_sample
        
    return real, synth

real, synth = inverse_scale_most_sim_vec(real_data, synth_data, features)

evaluator = TableEvaluator(real, synth)
evaluator.visual_evaluation()

for i in features:
    print(f'\nstatistical evaluation {i}')
    evaluator.evaluate(target_col=f'{i}', target_type='regr')
    

##############################################################################

# inverse scale all, check random vectors
def inverse_scale(real, synth, feature):

        if machine is not None:
            file = open(fr'00_data\03_train\{tunnel}\{tunnel}_{machine}_scaler_{feature}_{look_back}.pkl', 'rb')
        else:
            file = open(fr'00_data\03_train\{tunnel}\{tunnel}_scaler_{feature}_{look_back}.pkl', 'rb')
        scaler = pickle.load(file)
        file.close()

        real_inv = scaler.inverse_transform(real) # eval to interpret string as variable
        synth_inv = scaler.inverse_transform(synth)
        
        return real_inv, synth_inv

###############################################################################

# reshape, inverse scale and save as dataframe
real_pene_reshaped = real_data_pene.reshape(-1, 1)
real_torque_reshaped = real_data_torque.reshape(-1, 1)
real_adv_force_reshaped = real_data_adv_force.reshape(-1, 1)
real_ucs_reshaped = real_data_ucs.reshape(-1, 1)

synth_pene_reshaped = synth_data_pene.reshape(-1, 1)
synth_torque_reshaped = synth_data_torque.reshape(-1, 1)
synth_adv_force_reshaped = synth_data_adv_force.reshape(-1, 1)
synth_ucs_reshaped = synth_data_ucs.reshape(-1, 1)

real_data_pene_inv, synth_data_pene_inv = inverse_scale(
    real_pene_reshaped, synth_pene_reshaped, features[0])
real_data_adv_force_inv, synth_data_adv_force_inv = inverse_scale(
    real_adv_force_reshaped, synth_adv_force_reshaped, features[1])
real_data_torque_inv, synth_data_torque_inv = inverse_scale(
    real_torque_reshaped, synth_torque_reshaped, features[2])
real_data_ucs_inv, synth_data_ucs_inv = inverse_scale(
    real_ucs_reshaped, synth_ucs_reshaped, features[3])

real_df = pd.DataFrame({'Tunnel Distance [m]':np.arange(0, 51200, 0.05).astype(float),
                        'Penetration [mm/rot]':np.round(real_data_pene_inv.squeeze().astype(float), 3),
                        'Total advance force [kN]':np.round(real_data_adv_force_inv.squeeze().astype(float), 3),
                        'Torque cutterhead [MNm]':np.round(real_data_torque_inv.squeeze().astype(float), 3),
                        'UCS [MPa]':np.round(real_data_ucs_inv.squeeze().astype(float), 3)})

synth_df = pd.DataFrame({'Tunnel Distance [m]':np.arange(0, 51200, 0.05).astype(float),
                        'Penetration [mm/rot]':np.round(synth_data_pene_inv.squeeze().astype(float), 2),
                        'Total advance force [kN]':np.round(synth_data_adv_force_inv.squeeze().astype(float), 2),
                        'Torque cutterhead [MNm]':np.round(synth_data_torque_inv.squeeze().astype(float), 2),
                        'UCS [MPa]':np.round(synth_data_ucs_inv.squeeze().astype(float), 2)})

synth_df.to_parquet(f'{filepath}/03_generated_data/synth_df_{tunnel}_{look_back}')
real_df.to_parquet(f'{filepath}/03_generated_data/real_df_{tunnel}_{look_back}')

##############################################################################

# plot
# real_df = pd.read_parquet(f'{filepath}/03_generated_data/real_df_{tunnel}_{look_back}')
# synth_df = pd.read_parquet(f'{filepath}/03_generated_data/synth_df_{tunnel}_{look_back}')

def plot_cont(df, FROM, TO, WINDOW, origin):
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(12, 8))
        
        # plot ucs
        ax1.plot(df['Tunnel Distance [m]'].values,
                  df['UCS [MPa]'].values, color='grey', alpha=0.6, label='UCS [MPa]')
        ax1.plot(df['Tunnel Distance [m]'], 
                  df['UCS [MPa]'].rolling(window=WINDOW, center=True).mean().values,
                  color='black', linewidth=0.5)
        # ax1.axhline(y=df['UCS [MPa]'].values.mean() + 2*s.stdev(df['UCS [MPa]'].values),
        #             color='black', alpha=0.9, linestyle='--', linewidth=0.5, label='+/- 2σ')
        # ax1.axhline(y=df['UCS [MPa]'].values.mean() - 2*s.stdev(df['UCS [MPa]'].values),
        #             color='black', alpha=0.9, linestyle='--', linewidth=0.5)
        ax1.set_xlim(FROM, TO)
        ax1.set_ylim(0, 350)
        ax1.set_ylabel('UCS [MPa]', rotation=90)
        # ax1.set_yticks([100, 200, 300])
        ax1.grid(alpha=0.5)
        ax1.legend(loc=4, prop={'size': 10})        
        
        # plot Penetration
        ax2.plot(df['Tunnel Distance [m]'].values, df['Penetration [mm/rot]'].values, 
                 color='grey', alpha=0.6, label='Penetration [mm/rot]')
        ax2.plot(df['Tunnel Distance [m]'].values, df['Penetration [mm/rot]'].rolling(window=WINDOW, center=True).mean().values,
                 color='black', linewidth=0.5)
        # ax2.axhline(y=df['Penetration [mm/rot]'].mean() + 2*s.stdev(df['Penetration [mm/rot]']),
        #             color='black', alpha=0.9, linestyle='--', linewidth=0.5, label='+/- 2σ')
        # ax2.axhline(y=df['Penetration [mm/rot]'].mean() - 2*s.stdev(df['Penetration [mm/rot]']),
        #             color='black', alpha=0.9, linestyle='--', linewidth=0.5)
        ax2.set_xlim(FROM, TO)
        ax2.set_ylim(0, 30)
        #ax2.set_ylim(ax2.get_ylim()[::-1])
        
        ax2.set_ylabel('Penetration [mm/rot]', rotation=90)
        # ax2.set_yticks([0, 1])
        ax2.grid(alpha=0.5)
        ax2.legend(loc=4, prop={'size': 10})

        # plot torque
        ax3.plot(df['Tunnel Distance [m]'].values, df['Torque cutterhead [MNm]'].values,
                    color='grey', alpha=0.6, label='Torque cutterhead [MNm]')
        ax3.plot(df['Tunnel Distance [m]'].values,
                 df['Torque cutterhead [MNm]'].rolling(window=WINDOW, center=True).mean().values,
                 color='black', linewidth=0.5)
        # ax3.axhline(y=df['Torque cutterhead [MNm]'].values.mean() + 2*s.stdev(df['Torque cutterhead [MNm]'].values),
        #             color='black', alpha=0.9, linestyle='--', linewidth=0.5, label='+/- 2σ')
        # ax3.axhline(y=df['Torque cutterhead [MNm]'].values.mean() - 2*s.stdev(df['Torque cutterhead [MNm]'].values),
        #             color='black', alpha=0.9, linestyle='--', linewidth=0.5)     
        ax3.set_xlim(FROM, TO)
        ax3.set_ylim(0, 6)
        ax3.set_ylabel('Torque Cutterhead\n'
                       ' [MNm]', rotation=90)
        # ax3.set_yticks([0, 0.5, 1, 1.5])
        # ax3.set_xticklabels([])
        ax3.legend(loc=4, prop={'size': 10})
        ax3.grid(alpha=0.5)
        
        # plot advance force
        ax4.plot(df['Tunnel Distance [m]'].values, df['Total advance force [kN]'].values,
                    color='grey', alpha=0.6, label='Total advance force [kN]')
        ax4.plot(df['Tunnel Distance [m]'].values,
                 df['Total advance force [kN]'].rolling(window=WINDOW, center=True).mean().values,
                 color='black', linewidth=0.5)
        # ax4.axhline(y=df['Total advance force [kN]'].values.mean() + 2*s.stdev(df['Total advance force [kN]'].values),
        #             color='black', alpha=0.9, linestyle='--', linewidth=0.5, label='+/- 2σ')
        # ax4.axhline(y=df['Total advance force [kN]'].values.mean() - 2*s.stdev(df['Total advance force [kN]'].values),
        #             color='black', alpha=0.9, linestyle='--', linewidth=0.5)     
        ax4.set_xlim(FROM, TO)
        ax4.set_ylim(0, 30000)
        ax4.set_ylabel('Total advance force [kN]', rotation=90)
        # ax4.set_yticks([0, 0.5, 1, 1.5])
        # ax4.set_xticklabels([])
        ax4.legend(loc=4, prop={'size': 10})
        ax4.grid(alpha=0.5)
        
        # save fig
        plt.tight_layout()
        plt.savefig(f'{filepath}\{origin}_{FROM}_{TO}.png', dpi=300)


plot_cont(synth_df, 0, 2000, 25, 'synth')
plot_cont(real_df, 0, 2000, 25, 'real')
plot_cont(synth_df, 0, 200, 25, 'synth')
plot_cont(real_df, 0, 200, 25, 'real')