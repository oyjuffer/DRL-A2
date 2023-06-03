
# pip install gymnasium
# pip install "gymnasium[atari, accept-rom-license]"

# Libraries
import gymnasium as gym

# Game Loop
env = gym.make("ALE/MarioBros-v5", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()