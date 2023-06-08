# DRL - Assignment2
# Run a model trained on the SpaceInvadersNoFrameskip-v4 to play the game
# Yorick Juffer, Alejandro Sánchez Roncero
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.atari_wrappers import AtariWrapper 
from stable_baselines3.common.monitor import Monitor, ResultsWriter
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


env_id = 'SpaceInvadersNoFrameskip-v4'
case_id = "01"

model = PPO.load("./models/{}/best_model.zip".format(case_id), print_system_info=True)
env = model.get_env()

env = gym.make(env_id, render_mode="human") # difficulty: 0, mode: 0, 6 discrete actions

# Resize to a square image: 84x84 by default
# Grayscale observation
# Clip reward to {-1, 0, 1}
env = AtariWrapper(env, frame_skip=4, action_repeat_probability=0.0) 
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
env = VecFrameStack(env, n_stack=4)  # Stack 4 frames together, to give the agent a sense of motion
env = VecTransposeImage(env)  # Transpose the image to (color_channels, height, width)
env.metadata["render_fps"] = 30

episodes = 5

for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
        # env.render()
    print(f'Episode: {episode}, Score: {score}')
env.close()