import cv2
import gym
import numpy as np
from gym import ObservationWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

class PreprocessObservation(ObservationWrapper):
    '''
    Preprocesses simulator image inputs for CNN-use.
    '''
    def __init__(self, env, new_shape=(84, 84)):
        super().__init__(env)
        self.new_shape = new_shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(3, *new_shape), dtype=np.float32
        )

    def observation(self, obs):
        '''
        Preprocessing (automatically called by Gym).
        '''
        # obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB) # bgr to rgb conversion
        obs = cv2.resize(obs, self.new_shape) # image resizing to reduce input size and computation complexity
        obs = obs.astype(np.float32) / 255.0 # normalize pixel values (0-1)
        obs = np.transpose(obs, (2, 0, 1)) #  rearrange shape to (channels, height, width) for SB3-compatibility
        return obs

# class RewardLoggerCallback(BaseCallback):
#     '''
#     Custom callback function for keeping track of reward progress during training.
#     '''
#     def __init__(self, verbose=0):
#         super().__init__(verbose)
#         self.episode_rewards = []

#     def _on_step(self) -> bool:
#         # `infos` is a list of dicts (one per env in vectorized envs)
#         infos = self.locals.get('infos', [{}])
#         for info in infos:
#             # When an episode ends, info contains 'episode' key
#             if 'episode' in info:
#                 reward = info['episode']['r']  # episode reward
#                 self.episode_rewards.append(reward)
#                 print(f"Episode finished: reward={reward}")
#         return True


class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_dir="./logs", max_ep_len=1000, verbose=0):
        super().__init__(verbose)
        self.writer = SummaryWriter(log_dir)
        self.episode_rewards = []
        self.episode_num = 0
        self.step_count = 0
        self.max_ep_len = max_ep_len

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]

        if len(self.episode_rewards) <= self.episode_num:
            self.episode_rewards.append(reward)
        else:
            self.episode_rewards[self.episode_num] += reward

        self.step_count += 1
        if done or self.step_count >= self.max_ep_len:
            self.writer.add_scalar("reward/episode_reward", self.episode_rewards[self.episode_num], self.episode_num)
            print(f"Episode {self.episode_num} reward: {self.episode_rewards[self.episode_num]}")
            self.episode_num += 1
            self.step_count = 0
            self.writer.flush()  # <-- this line ensures TensorBoard logs get written to disk

        return True
