# TRAIN
# trains a DRL model to play mario bros using PPO. 

import os
import gymnasium as gym
from stable_baselines3 import PPO

# globals
timesteps = 10000       # will save model after n timesteps
max = 50                # max * timesteps = max timesteps before exit

# create dir for saving models
models_dir = "models/PPO"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# create dir for tensor logs
logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)

### TRAINING THE MODEL ###
env = gym.make("ALE/MarioBros-v5")
env.reset()
model = PPO('CnnPolicy', 
            env, 
            learning_rate=1e-05,
            # batch_size=32,
            # n_epochs=20,
            # gamma=0.99,
            # clip_range=0.2,
            # target_kl=0.01,
            verbose=1, 
            tensorboard_log=logdir)

for i in range(max):
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{timesteps*(i+1)}")
