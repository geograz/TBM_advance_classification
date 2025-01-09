# -*- coding: utf-8 -*-
"""
Challenges and Opportunities of Data Driven Advance Classification for Hard
Rock TBM excavations

---- script to paper
DOI: XXXXXXX

Wasserstein Generative Adversarial Network Training Function

@author: Paul Unterlass
Created on Thu Jun 13 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
# from WGAN_02_model_8192 import Discriminator, Generator, initialize_weights
from WGAN_02_model_4096 import Discriminator, Generator, initialize_weights

# check if gpu is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    cpu = torch.device("cpu")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the CPU")
    
# =============================================================================
# Hyperparameters etc.
# =============================================================================

LEARNING_RATE = 5e-5
BATCH_SIZE = 32
CHANNEL = 4 # number of features
Z_DIM = 100 # noise dimension
NUM_EPOCHS = 25
FEATURES_CRITIC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 3 # for every 3 update steps in critic, one update step in generator
WEIGHT_CLIP = 0.01

tunnel = 'TBM_A'
FEATURE_NAME1 = 'pene'
FEATURE_NAME2 = 'adv_force'
FEATURE_NAME3 = 'torque'
FEATURE_NAME4 = 'ucs'
look_back = 4096 # 8192
machine = None # 'S980'

# =============================================================================
# data and initalization
# =============================================================================

if machine is not None:
    train_X = np.load(fr'00_data\03_train\{tunnel}\{tunnel}_train_X_{FEATURE_NAME1}_{FEATURE_NAME2}_{FEATURE_NAME3}_{FEATURE_NAME4}_{look_back}_{machine}.npy')
else:
    train_X = np.load(fr'00_data\03_train\{tunnel}\{tunnel}_train_X_{FEATURE_NAME1}_{FEATURE_NAME2}_{FEATURE_NAME3}_{FEATURE_NAME4}_{look_back}.npy')

train_X_tens = torch.from_numpy(train_X).float()
loader = DataLoader(train_X_tens, batch_size=BATCH_SIZE, shuffle=True)

# initialize gen and disc/critic
gen = Generator(Z_DIM, CHANNEL, FEATURES_GEN).to(device)
critic = Discriminator(CHANNEL, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)

# initializate optimizer
opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
opt_critic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)

# =============================================================================
# training process
# =============================================================================
time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
step = 0

gen.train()
critic.train()

critic_losses = []
generator_losses = []

for epoch in range(NUM_EPOCHS):

    for batch_idx, (data) in enumerate(loader):
        data = data.to(device)
        cur_batch_size = data.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        for _ in range(CRITIC_ITERATIONS):
            noise = np.random.uniform(0, 1, size=(cur_batch_size, int(Z_DIM), 1))
            noise = torch.from_numpy(noise).float().to(device)
            fake = gen(noise)
            critic_real = critic(data).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            critic.zero_grad()
            loss_critic.backward(retain_graph=True) # because reutilisation of fake in train Gen
            opt_critic.step()
            
            # clip critic weights between -0.01, 0.01
            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
        # Print losses occasionally
        if batch_idx % 500 == 0 and batch_idx > 0:
            gen.eval()
            critic.eval()
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )
                
            critic_losses.append(loss_critic.to(cpu).detach().numpy()) # save losses
            generator_losses.append(loss_gen.to(cpu).detach().numpy())
                                    
            step += 1
            gen.train()
            critic.train()
            
    # save model
    if epoch == NUM_EPOCHS-1:
        torch.save(gen.state_dict(),
                   fr'02_models/{tunnel}_gen_state_{time}_bs{BATCH_SIZE}_gen{step+1}.h5')
        torch.save(gen,
                   fr'02_models/{tunnel}_gen_entire_{time}_bs{BATCH_SIZE}_gen{step+1}.h5')

# =============================================================================
# visualize training process
# =============================================================================
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 9))
x = range(step)
ax1.plot(x, generator_losses, color='C0', linestyle='-', label='gen loss', linewidth=0.3)
ax1.plot(x, critic_losses, color='C1', linestyle='-', label='critic loss', linewidth=0.3)
ax1.set_ylabel('loss')
ax1.legend()
ax1.grid(alpha=0.5)

plt.tight_layout()
plt.savefig(fr'01_results/{tunnel}_training{step}.jpg', dpi=300)