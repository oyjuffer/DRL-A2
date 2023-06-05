# RUN
# A model from a certain timestamp is used to play mario bros which can be followed in a game window. 

import gymnasium as gym
from stable_baselines3 import PPO

models_dir = "models/PPO"
env = gym.make("ALE/MarioBros-v5", render_mode="human")
model_path = f"{models_dir}/230000.zip"
model = PPO.load(model_path, env=env)

# sets the model to play the game
env.metadata["render_fps"] = 30
vec_env = model.get_env()
obs = vec_env.reset()
while True:
    action, _state = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")

env.close()