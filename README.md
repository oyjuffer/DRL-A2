# DRL-A2
 
The train.py will train a DRL PPO model on the mario game.\
The run.py will display the resulting model at a certain timestamp via a game window. \

INSTALLS - These will install gym but also the game ROM. \
pip install gymnasium \
pip install "gymnasium[atari, accept-rom-license]" \
pip install gym[mujoco]


RESOURCES: \

ENVIRONMENT \
https://gymnasium.farama.org/environments/atari/ \
https://gymnasium.farama.org/environments/atari/mario_bros/ \

STABLE BASELINE 3 \
https://stable-baselines3.readthedocs.io/en/master/guide/install.html \
https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html \

PPO \
https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html \
https://stable-baselines3.readthedocs.io/en/v1.8.0/guide/custom_policy.html# \

GUIDE \
https://pythonprogramming.net/saving-and-loading-reinforcement-learning-stable-baselines-3-tutorial/ \

HYPERPARAMETERS \
https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe \
https://github.com/llSourcell/Unity_ML_Agents/blob/master/docs/best-practices-ppo.md \

TENSORBOARD \
tensorboard --logdir=logs \