# DRL - Assignment2
# Run a model trained on the SpaceInvadersNoFrameskip-v4 to play the game
# Yorick Juffer, Alejandro Sánchez Roncero

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback
from IPython import display
import matplotlib
import matplotlib.pyplot as plt


env_id = 'SpaceInvadersNoFrameskip-v4'

model = PPO.load("ppo_space_invaders", print_system_info=True)
env = model.get_env()

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