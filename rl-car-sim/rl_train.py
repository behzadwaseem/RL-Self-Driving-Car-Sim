import gym
import numpy as np

# DonkeyCar Gym environment
import gym_donkeycar
# from gym.wrappers import RecordEpisodeStatistics
from gym.wrappers import Monitor
TRAINING_RUN = 1

# RL algorithm from Stable-Baselines3
from stable_baselines3 import SAC

# Wrappers for Gym environments
from stable_baselines3.common.vec_env import DummyVecEnv

# Your custom preprocessing wrapper & donkey env
from wrappers import PreprocessObservation, RewardLoggerCallback
from myconfig import MY_DONKEY_GYM_ENV_NAME


# CREATE AND WRAP ENVIRONMENT:

# Create the base environment
env = gym.make(MY_DONKEY_GYM_ENV_NAME)
# env = RecordEpisodeStatistics(env)

# Preprocess observations (convert image to 84x84 RGB with channels first)
env = PreprocessObservation(env)
# env = Monitor(env, f"./monitor_logs_{TRAINING_RUN}/")
obs = env.reset()
# verify obs variables:
print(type(obs))         # Should be <class 'numpy.ndarray'>
print(obs.shape)         # Should be something like (120, 160, 3)
print(obs.dtype)         # Should be uint8
print(obs[0][0])         # Should show pixel values like [57 81 123]

print("Initial observation shape:", obs.shape)

# Test movement
obs, reward, done, info = env.step([0.0, 0.3])  # try moving forward
print("First step — reward:", reward)

# Vectorize the environment (SB3 requires this, even for single envs)
env = DummyVecEnv([lambda: env])

# DEFINE SAC AGENT:

# Create SAC model with CNN policy (for image input)
model = SAC(
    policy="CnnPolicy",
    env=env,
    verbose=1,                  # print training info
    buffer_size=20_000,        # size of replay buffer
    learning_rate=3e-4,         # how fast it learns
    train_freq=1,               # how often it trains
    gradient_steps=1,           # number of updates per train step
    batch_size=64,              # mini-batch size for training
    tensorboard_log="./logs",    # optional for visual logs
    policy_kwargs={"normalize_images": False}, # don't normalize images again (we already do this in PreprocessObservation wrapper)

)
# TRAIN AGENT:

# Train for 100k steps — should take about 1–2 hours on M1 Mac
TIMESTEPS = 100_000
callback = RewardLoggerCallback(log_dir="./logs")

# Start training
print("start training loop . . .")
model.learn(total_timesteps=TIMESTEPS, 
            callback=callback
            )
# model.learn(total_timesteps=TIMESTEPS)

# SAVE TRAINED MODEL:
model.save("sac_donkeycar_model")
