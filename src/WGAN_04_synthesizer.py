# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 2024

@author: Paul Unterlass

Function to synthesize n sequences of synthetic data using a trained generator
neural network
"""

import torch
import numpy as np

# check if gpu is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    cpu = torch.device("cpu")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the CPU")
    
# =============================================================================
# load model and real data
# =============================================================================

tunnel = 'BBT'
FEATURE_NAME1 = 'pene'
FEATURE_NAME2 = 'adv_force'
FEATURE_NAME3 = 'torque'
FEATURE_NAME4 = 'ucs'
machine = None

# filepath = r'C:/02_Research/01_Unterlass/01_GANs/03_GANs/06_WGAN_oenorm/01_results/BBT_2024-06-11_18-21-48_bs32_gen171_pene_adv_force_torque_ucs_8192'
filepath = r'C:/02_Research/01_Unterlass/01_GANs/03_GANs/06_WGAN_oenorm/01_results/BBT_2024-06-13_15-07-47_bs32_gen426_pene_adv_force_torque_ucs_4096'

# load model
gen = torch.load(r'02_models\BBT_gen_entire_2024-06-13_15-07-47_bs32_gen426.h5')

# =============================================================================
# snythetize data
# =============================================================================

SEQ_LENGTH = 4096 # lenght of vectors
BATCH_SIZE = 32
N_PASSES = 2500 # number of vectors to synthesize
Z_DIM = 100 # noise dimension
CHANNEL = 4 # number of features to synthesize
run = 1 # running number

synth_data = []

for i in range(N_PASSES):
    noise = np.random.uniform(0, 1, size=(BATCH_SIZE, int(Z_DIM), 1))
    noise = torch.from_numpy(noise).float().to(device)
    synth_sample = gen(noise).to(cpu).detach().numpy()
    synth_sample = synth_sample[np.random.randint(0, synth_sample.shape[0],
                                         size=1)][0]   
    synth_data.append(synth_sample)

synth_data = np.stack(synth_data)
print(synth_data.shape)

np.save(fr'{filepath}/03_generated_data/{tunnel}_length{SEQ_LENGTH}_n{N_PASSES}_{FEATURE_NAME1}_{FEATURE_NAME2}_{FEATURE_NAME3}_{FEATURE_NAME4}_{run}.npy', synth_data)
