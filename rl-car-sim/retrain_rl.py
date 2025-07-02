# resume_training.py

import gym
import gym_donkeycar
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from wrappers import PreprocessObservation, RewardLoggerCallback
from myconfig import MY_DONKEY_GYM_ENV_NAME

# recreate environment:
env = gym.make(MY_DONKEY_GYM_ENV_NAME)
env = PreprocessObservation(env)
env = DummyVecEnv([lambda: env])

# load model:
model = SAC.load("sac_donkeycar_model.zip", env=env)
callback = RewardLoggerCallback(log_dir="./logs")

# resume training:
model.learn(total_timesteps=200_000, reset_num_timesteps=False, callback=callback)

# save updated model:
model.save("sac_donkeycar_model_completed")
