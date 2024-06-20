"""
Created on Thu Jun 13 2024

@author: Paul Unterlass / Georg Erharter
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pickle
from table_evaluator import TableEvaluator
import pandas as pd

N_VECTORS = 2500 # number of vectors of generated data
VECTOR_LENGTH = 4096  # length of each vector
tunnel = 'TBM_A' # 'TBM_B' # TBM_C
FEATURE_NAME1 = 'pene'
FEATURE_NAME2 = 'adv_force'
FEATURE_NAME3 = 'torque'
FEATURE_NAME4 = 'ucs'
features = [FEATURE_NAME1, FEATURE_NAME2, FEATURE_NAME3, FEATURE_NAME4]
machine = None
look_back = 4096
N_PASSES = 2500
run = 1

filepath = r'C:/02_Research/01_Unterlass/01_GANs/03_GANs/06_WGAN_oenorm/01_results/TBM_A_2024-06-13_15-07-47_bs32_gen426_pene_adv_force_torque_ucs_4096'
# filepath = r'C:/02_Research/01_Unterlass/01_GANs/03_GANs/06_WGAN_oenorm/01_results/TBM_C_2024-06-13_15-27-09_bs32_gen301_pene_adv_force_torque_ucs_4096'
# filepath = r'C:/02_Research/01_Unterlass/01_GANs/03_GANs/06_WGAN_oenorm/01_results/TBM_B_2024-06-13_14-07-27_bs32_gen226__pene_adv_force_torque_ucs_4096'

# =============================================================================
# import data
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
# real_data = real_data[:N_VECTORS]

# split real data per feature and drop every 100th vector. Two consecutive
# vectors differ only by one digit due to the sliding window approach. To keep
# computational costs low but still get varying vectors every 100th vector
# is dropped from the real data vectors

real_data_pene = real_data[:, 0, :]
real_data_pene = real_data_pene[::100]
real_data_adv_force = real_data[:, 1, :]
real_data_adv_force = real_data_adv_force[::100]
real_data_torque = real_data[:, 2, :]
real_data_torque = real_data_torque[::100]
real_data_ucs = real_data[:, 3, :]
real_data_ucs = real_data_ucs[::100]

synth_data_pene = synth_data[:, 0, :]
synth_data_adv_force = synth_data[:, 1, :]
synth_data_torque = synth_data[:, 2, :]
synth_data_ucs = synth_data[:, 3, :]

# =============================================================================
# compute and collect the distances from all generated vectors to all training
# vectors per feature
# =============================================================================

def compute_vectors(real_data, synth_data):
    similar_vectors = []
    min_distances = []
    
    disimilar_vectors = []
    max_distances = []
    
    for i in range(len(synth_data)): 
        if i % 500 == 0:
            print(f'{i} of {N_VECTORS}')
        distances = np.linalg.norm(synth_data[i] - real_data, axis=1)
        similar_vectors.append(np.argmin(distances))
        min_distances.append(distances[np.argmin(distances)])
        disimilar_vectors.append(np.argmax(distances))
        max_distances.append(distances[np.argmax(distances)])
    return similar_vectors, disimilar_vectors

'''
# vectorized numpy way, if computer has enough memory
def compute_vectors(real_data, synth_data, batch_size=200):
    similar_vectors = []
    min_distances = []
    disimilar_vectors = []
    max_distances = []
    
    num_batches = int(np.ceil(len(synth_data) / batch_size))
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(synth_data))
        
        batch = synth_data[start_idx:end_idx]
        
        # Compute distances for the current batch
        distances = np.linalg.norm(batch[:, np.newaxis] - real_data, axis=2)
        
        # Find indices and values of minimum and maximum distances
        similar_vectors.extend(np.argmin(distances, axis=1))
        min_distances.extend(np.min(distances, axis=1))
        
        disimilar_vectors.extend(np.argmax(distances, axis=1))
        max_distances.extend(np.max(distances, axis=1))
        
        print(f'Processed batch {batch_idx + 1} of {num_batches}')
    
    return similar_vectors, disimilar_vectors
'''

similar_vectors_pene, disimilar_vectors_pene = compute_vectors(
    real_data_pene, synth_data_pene)

similar_vectors_adv_force, disimilar_vectors_adv_force = compute_vectors(
    real_data_adv_force, synth_data_adv_force)

similar_vectors_torque, disimilar_vectors_torque = compute_vectors(
    real_data_torque, synth_data_torque)

similar_vectors_ucs, disimilar_vectors_ucs = compute_vectors(
    real_data_ucs, synth_data_ucs)

# =============================================================================
# inverse scale real and synth data per feature
# =============================================================================

def inverse_scale(real, synth, feature):
    if machine is not None:
        file = open(fr'00_data\03_train\{tunnel}\{tunnel}_{machine}_scaler_{feature}_{look_back}.pkl', 'rb')
    else:
        file = open(fr'00_data\03_train\{tunnel}\{tunnel}_scaler_{feature}_{look_back}.pkl', 'rb')
    scaler = pickle.load(file)
    file.close()
    
    real_inv = scaler.inverse_transform(real)
    synth_inv = scaler.inverse_transform(synth)
    
    return real_inv, synth_inv

real_data_pene_inv, synth_data_pene_inv = inverse_scale(
    real_data_pene, synth_data_pene, features[0])
real_data_adv_force_inv, synth_data_adv_force_inv = inverse_scale(
    real_data_adv_force, synth_data_adv_force, features[1])
real_data_torque_inv, synth_data_torque_inv = inverse_scale(
    real_data_torque, synth_data_torque, features[2])
real_data_ucs_inv, synth_data_ucs_inv = inverse_scale(
    real_data_ucs, synth_data_ucs, features[3])

real_np = np.stack((real_data_pene_inv, real_data_adv_force_inv,
                      real_data_torque_inv, real_data_ucs_inv), axis=1)
synth_np = np.stack((synth_data_pene_inv, synth_data_adv_force_inv,
                       synth_data_torque_inv, synth_data_ucs_inv), axis=1)

# np.save(fr'{filepath}\03_generated_data\{tunnel}_synth_pene_adv_force_torque_ucs_{look_back}.npy', synth_np)

# =============================================================================
# save vectors/distances and plot most sim vectors
# =============================================================================

# if False saves vectors/distances and plots
# if True only plots vectors

features_full = ['Penetration [mm/rot]', 'Total advance force [kN]',
                 'Torque cutterhead [MNm]', 'UCS [MPa]']
only_plot = True

for i in range(len(features)):
    # load similar/disimilar vectors
    if only_plot == True:
        with open(f'{filepath}/04_vector_distance/similar_vectors_{features[i]}', 'rb') as f:
            similar_vectors = pickle.load(f)
            
        with open(f'{filepath}/04_vector_distance/disimilar_vectors_{features[i]}', 'rb') as f:
            disimilar_vectors = pickle.load(f)
    # save similar/disimilar vectors
    else:
        with open(f'{filepath}/04_vector_distance/similar_vectors_{features[i]}', 'wb') as f:
            pickle.dump(eval(f'similar_vectors_{features[i]}'), f)
    
        with open(f'{filepath}/04_vector_distance/disimilar_vectors_{features[i]}', 'wb') as f:
            pickle.dump(eval(f'disimilar_vectors_{features[i]}'), f)
            
        similar_vectors = eval(f'similar_vectors_{features[i]}')
        disimilar_vectors = eval(f'disimilar_vectors_{features[i]}')
    
    ###########################################################################
    # generate x sample plots
    
    # load scaler
    if machine is not None:
        file = open(fr'00_data\03_train\{tunnel}\{tunnel}_{machine}_scaler_{features[i]}_{look_back}.pkl', 'rb')
    else:
        file = open(fr'00_data\03_train\{tunnel}\{tunnel}_scaler_{features[i]}_{look_back}.pkl', 'rb')
    scaler = pickle.load(file)
    file.close()

    # inverse transform scaled data
    real_data = scaler.inverse_transform(eval(f'real_data_{features[i]}'))
    synth_data = scaler.inverse_transform(eval(f'synth_data_{features[i]}'))
    
    np.random.seed(8)
    for ii in np.random.choice(np.arange(len(synth_data)), size=2, replace=False):
        
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(range(VECTOR_LENGTH), synth_data[ii],
                label=f'generated data {features_full[i]}', color='black', lw=1.2,
                alpha=1, zorder=10)
        
        ax.plot(range(VECTOR_LENGTH), real_data[similar_vectors[ii]],
                label=f'most similar real data {features_full[i]}', color='C0',
                lw=1.2, linestyle='-', alpha=1, zorder=10)
        
        ax.plot(range(VECTOR_LENGTH), real_data[disimilar_vectors[ii]],
                label=f'most disimilar real data {features_full[i]}', color='C1',
                alpha=1, linestyle='-', zorder=0)
        
        ax.plot(range(VECTOR_LENGTH), synth_data[ii], range(VECTOR_LENGTH),
                real_data[similar_vectors[ii]], label='_nolegend_',
                color='black', lw=2, alpha=0)
        
        ax.fill_between(range(VECTOR_LENGTH), synth_data[ii],
                        real_data[similar_vectors[ii]], facecolor='lightgray',
                        interpolate=True)
        #ax.patch.set_facecolor('grey')
        #ax.patch.set_alpha(0.7)
        plt.xlim(0,VECTOR_LENGTH)
        plt.ylabel(f'{features_full[i]}')
        ax.grid(alpha=0.6, linewidth=0.4)
        ax.legend(loc=2, fancybox=True)
        
        plt.tight_layout()
        plt.savefig(f'{filepath}/04_vector_distance/01_plots/filled_line_{features[i]}_{ii}',
                    dpi=300)

        # single plots
        fig = plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(3, 1)
        ax1 = plt.subplot(gs[0])
        ax1.plot(range(VECTOR_LENGTH), synth_data[ii],
                 label=f'generated data {features_full[i]}', color='black', lw=1)
        plt.xlim(0,VECTOR_LENGTH)
        #plt.ylim(0,1)
        ax2 = plt.subplot(gs[1])
        ax2.plot(range(VECTOR_LENGTH), real_data[similar_vectors[ii]],
                 label=f'most similar real data {features_full[i]}', color='C0',
                 lw=1)
        plt.xlim(0,VECTOR_LENGTH)
        plt.ylabel(f'{features_full[i]}')
        #plt.ylim(0,1)
        ax3 = plt.subplot(gs[2])
        ax3.plot(range(VECTOR_LENGTH), real_data[disimilar_vectors[ii]],
                 label=f'most disimilar real data {features_full[i]}', color='C1',
                 lw=1)
        plt.xlim(0,VECTOR_LENGTH)
        #plt.ylim(0,1)
        ax1.grid(alpha=0.5)
        ax1.legend(loc=2)
        ax2.grid(alpha=0.5)
        ax2.legend(loc=2)
        ax3.grid(alpha=0.5)
        ax3.legend(loc=2)
        
        plt.tight_layout()
        plt.savefig(f'{filepath}/04_vector_distance/01_plots/single_plots_{features[i]}_{ii}',
            dpi=300)

        # histograms
        fig = plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(1, 3) #, width_ratios=[1, 1]
        ax1 = plt.subplot(gs[0])
        ax1.hist(synth_data[ii], bins=30, range=(real_data.min(), real_data.max()),
                 color='black', edgecolor='white',
                 label=f'generated data {features_full[i]}')
        ax2 = plt.subplot(gs[1])
        ax2.hist(real_data[similar_vectors[ii]],
                 range=(real_data.min(), real_data.max()),
                 bins=30, color='C0', edgecolor='black',
                 label=f'most similar real data {features_full[i]}')
        ax3 = plt.subplot(gs[2])
        ax3.hist(real_data[disimilar_vectors[ii]],
                 range=(real_data.min(), real_data.max()),
                 bins=30, color='C1', edgecolor='black',
                 label=f'most disimilar real data {features_full[i]}')
        ax1.legend(loc=2)
        ax2.legend(loc=2)
        ax3.legend(loc=2)
        
        plt.tight_layout()
        plt.savefig(f'{filepath}/04_vector_distance/01_plots/histograms_{features[i]}_{ii}',
                    dpi=300)
        
        # single plots + histograms
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(3, 2, width_ratios=[4, 1])
        ax1 = plt.subplot(gs[0])
        ax1.plot(range(VECTOR_LENGTH), synth_data[ii],
                 label=f'generated data {features_full[i]}', color='black', lw=1)
        plt.xlim(0,VECTOR_LENGTH)
        #plt.ylim(0,1)
        ax2 = plt.subplot(gs[1])
        ax2.hist(synth_data[ii], bins=30,
                 range=(real_data.min(), real_data.max()),
                 color='black', edgecolor='white')
        ax3 = plt.subplot(gs[2])
        ax3.plot(range(VECTOR_LENGTH), real_data[similar_vectors[ii]],
                 label=f'most similar real data {features_full[i]}', color='C0',
                 lw=1)
        plt.xlim(0,VECTOR_LENGTH)
        plt.ylabel(f'{features[i]}')
        #plt.ylim(0,1)
        ax4 = plt.subplot(gs[3])
        ax4.hist(real_data[similar_vectors[ii]],
                 range=(real_data.min(), real_data.max()),
                 bins=30, color='C0', edgecolor='black',
                 label=f'most similar real data {features_full[i]}')
        ax5 = plt.subplot(gs[4])
        ax5.plot(range(VECTOR_LENGTH), real_data[disimilar_vectors[ii]],
                 label=f'most disimilar real data {features_full[i]}', color='C1',
                 lw=1)
        plt.xlim(0,VECTOR_LENGTH)
        #plt.ylim(0,1)
        ax6 = plt.subplot(gs[5])
        ax6.hist(real_data[disimilar_vectors[ii]],
                 range=(real_data.min(), real_data.max()),
                 bins=30, color='C1', edgecolor='black',
                 label=f'most disimilar real data {features_full[i]}')
        ax1.grid(alpha=0.5)
        ax1.legend(loc=3)
        # ax2.grid(alpha=0.5)
        ax3.grid(alpha=0.5)
        ax3.legend(loc=3)
        # ax4.grid(alpha=0.5)
        ax5.grid(alpha=0.5)
        ax5.legend(loc=3)
        # ax6.grid(alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{filepath}/04_vector_distance/01_plots/plots_hist_{features[i]}_{ii}',
                    dpi=300)
        
# =============================================================================
# table evaluator package to evaluate similarity of real and synthetic data
# after: https://baukebrenninkmeijer.github.io/table-evaluator/
# =============================================================================

for i in range(len(features)):
    
    # load similar/disimilar vectors
    with open(f'{filepath}/04_vector_distance/similar_vectors_{features[i]}', 'rb') as f:
        similar_vectors = pickle.load(f)
        
    with open(f'{filepath}/04_vector_distance/disimilar_vectors_{features[i]}', 'rb') as f:
        disimilar_vectors = pickle.load(f)
    
    # load scalers            
    if machine is not None:
        file = open(fr'00_data\03_train\{tunnel}\{tunnel}_{machine}_scaler_{features[i]}_{look_back}.pkl', 'rb')
    else:
        file = open(fr'00_data\03_train\{tunnel}\{tunnel}_scaler_{features[i]}_{look_back}.pkl', 'rb')
    scaler = pickle.load(file)
    file.close()

    # inverse transform scaled data
    real_data = scaler.inverse_transform(eval(f'real_data_{features[i]}'))
    synth_data = scaler.inverse_transform(eval(f'synth_data_{features[i]}'))
    
    # calculate statistical similarity and visualize similarity
    # of two randomly choosen vectors of synthetic and real data
    
    np.random.seed(1)
    for ii in np.random.choice(np.arange(len(real_data)), size=1, replace=False):    
        idx = ii
        
        real_data_sample = real_np[idx].transpose()
        synth_data_sample = synth_np[similar_vectors[idx]].transpose()
        real = pd.DataFrame(real_data_sample, columns=features)
        synth = pd.DataFrame(synth_data_sample, columns=features)
        
        # plots for visual evaluation
        evaluator = TableEvaluator(real, synth)
        # evaluator.visual_evaluation()
        
        # statistical evaluation
        print(f'{features_full[i]}:\nindex real: {idx}\nindex most similar synthetic: {similar_vectors[idx]}')
        evaluator.evaluate(target_col=f'{features[i]}', target_type='regr')
                
# =============================================================================
# black & white
# =============================================================================

# for i in np.random.choice(np.arange(N_VECTORS), size=2, replace=False):
    
#     fig, ax = plt.subplots(figsize=(15, 5))
#     ax.plot(range(VECTOR_LENGTH), synth_data[i],
#             label='generated data', color='black', lw=1.2, alpha=1, zorder=10)
    
#     ax.plot(range(VECTOR_LENGTH), real_data[similar_vectors[i]],
#             label='most similar real data', color='white', lw=1.2,
#             linestyle='-', alpha=1, zorder=10)
    
#     ax.plot(range(VECTOR_LENGTH), real_data[disimilar_vectors[i]],
#             label='most disimilar real data', color='white', alpha=0.6,
#             linestyle=':', zorder=0)
    
#     ax.plot(range(VECTOR_LENGTH), synth_data[i], range(VECTOR_LENGTH), real_data[similar_vectors[i]],
#             label='_nolegend_', color='black', lw=2, alpha=0)
    
#     ax.fill_between(range(VECTOR_LENGTH), synth_data[i], real_data[similar_vectors[i]], facecolor='black', interpolate=True)
#     ax.patch.set_facecolor('grey')
#     ax.patch.set_alpha(0.7)
#     plt.xlim(0,VECTOR_LENGTH)
#     ax.grid(color='white', alpha=0.6, linewidth=0.4)
#     ax.legend(facecolor='lightgrey', loc=2, fancybox=True)
    
#     plt.tight_layout()
    
#     # single plots + histograms
    
#     fig = plt.figure(figsize=(12, 6))
#     gs = gridspec.GridSpec(3, 2, width_ratios=[4, 1])
    
#     ax1 = plt.subplot(gs[0])
#     ax1.plot(range(VECTOR_LENGTH), synth_data[i],
#               label='generated data', color='black', lw=1)
#     plt.xlim(0,VECTOR_LENGTH)

#     ax2 = plt.subplot(gs[1])
#     ax2.hist(synth_data[i], bins=30, range=(0,350), color='black',
#               edgecolor='white')
    
#     ax3 = plt.subplot(gs[2])
#     ax3.plot(range(VECTOR_LENGTH), real_data[similar_vectors[i]],
#               label='most similar real data', color='black', lw=1)
#     plt.xlim(0,VECTOR_LENGTH)
    
#     ax4 = plt.subplot(gs[3])
#     ax4.hist(real_data[similar_vectors[i]], range=(0,350), bins=30, color='black',
#               edgecolor='white', label='most similar real data')
    
#     ax5 = plt.subplot(gs[4])
#     ax5.plot(range(VECTOR_LENGTH), real_data[disimilar_vectors[i]],
#               label='most disimilar real data', color='black', lw=1)
#     plt.xlim(0,VECTOR_LENGTH)

#     ax6 = plt.subplot(gs[5])
#     ax6.hist(real_data[disimilar_vectors[i]], range=(0,350), bins=30, color='black',
#               edgecolor='white', label='most disimilar real data')
    
#     ax1.grid(alpha=0.5)
#     ax1.legend(loc=2)
#     ax2.grid(alpha=0.5)
#     ax3.grid(alpha=0.5)
#     ax3.legend(loc=2)
#     ax4.grid(alpha=0.5)
#     ax5.grid(alpha=0.5)
#     ax5.legend(loc=2)
#     ax6.grid(alpha=0.5)
    
#     plt.tight_layout()
    
