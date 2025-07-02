import gym
import gym_donkeycar
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from myconfig import MY_DONKEY_GYM_ENV_NAME
from wrappers import PreprocessObservation  # same wrapper you used for training

# LOAD AND WRAP ENVIRONMENT
env = gym.make(MY_DONKEY_GYM_ENV_NAME)
env = PreprocessObservation(env)
env = DummyVecEnv([lambda: env])

# LOAD TRAINED MODEL:
model = SAC.load("sac_donkeycar_model.zip", env=env)

# RUN ONE EPISODE
obs = env.reset()
done = False
total_reward = 0.0

while not done:
    action, _ = model.predict(obs, deterministic=True)  # deterministic for testing
    obs, reward, done, info = env.step(action)
    total_reward += reward
    env.render()  # Optional: call render if implemented; many gym envs skip this

print(f"Total reward this episode: {total_reward}")
env.close()