
# INSTALLS
# pip install gymnasium
# pip install "gymnasium[atari, accept-rom-license]"


# RESOURCES
# https://gymnasium.farama.org/environments/atari/
# https://gymnasium.farama.org/environments/atari/mario_bros/

# https://stable-baselines3.readthedocs.io/en/master/guide/install.html
# https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html

### LIBARIES ###
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import A2C

### TRAINING THE MODEL ###
env = gym.make("ALE/MarioBros-v5", render_mode="human")
model = A2C("CnnPolicy", env, learning_rate=0.0001, verbose=1)
model.learn(total_timesteps=1000)

### TESTING THE MODEL ###
env.metadata["render_fps"] = 30
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")