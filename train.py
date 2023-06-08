# DRL - Assignment2
# Train a model on the SpaceInvadersNoFrameskip-v4 
# Yorick Juffer, Alejandro Sánchez Roncero

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.atari_wrappers import AtariWrapper 
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt


# Hyperparameters
env_id = 'SpaceInvadersNoFrameskip-v4'
model_id = "ppo_space_invaders"
total_timesteps = 1e6
eval_freq = 50000


# Environment:
# - "rgb" -> observation_space=Box(0, 255, (width=210, height=160, channels=3), np.uint8)
# - frameskip=1
# - repeat_prob=0

# Create the environment
env = gym.make(env_id, render_mode="human") # difficulty: 0, mode: 0, 6 discrete actions
env = AtariWrapper(env, frame_skip=1)  # This applies a few additional wrappers
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
env = VecFrameStack(env, n_stack=4)  # Stack 4 frames together, to give the agent a sense of motion
env = VecTransposeImage(env)  # Transpose the image to (color_channels, height, width)

# CNN is more suitable here since Atary games have image-bases observations
model = PPO("CnnPolicy", env, verbose=1)

# Train the agent
callback = EvalCallback(env, best_model_save_path='./models/', log_path='./logs/', 
                        eval_freq=eval_freq,
                        deterministic=True, 
                        render=False
                        
                        )
model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)


model.save(model_id)