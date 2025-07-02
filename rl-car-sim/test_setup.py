# from gym_donkeycar.envs.donkey_env import DonkeyEnv
import gym_donkeycar
import gym
import cv2
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
from myconfig import DONKEY_GYM_ENV_NAME
from wrappers import PreprocessObservation

# generate env:
env = gym.make(DONKEY_GYM_ENV_NAME)
env = PreprocessObservation(env)
obs = env.reset()

# verify obs variables:
print(type(obs))         # Should be <class 'numpy.ndarray'>
print(obs.shape)         # Should be something like (120, 160, 3)
print(obs.dtype)         # Should be uint8
print(obs[0][0])         # Should show pixel values like [57 81 123]

# verify cv2 frame capture:
# resized = cv2.resize(obs, (640, 480), interpolation=cv2.INTER_NEAREST)
# plt.imshow(resized)
# plt.imshow(obs)
# plt.title("DonkeyCar Raw Image")
# plt.axis('off')
# plt.show()

# verify frames update with movement:
for i in range(5):
    print(f"Step: {i}")
    obs, reward, done, infos = env.step([0.0, 0.2])  # move forward
    plt.imshow(np.transpose(obs, (1, 2, 0)))
    plt.title("DonkeyCar Raw Image")
    plt.axis('off')
    plt.show()
    # time.sleep(5)

    print(f"Step {i}: shape={obs.shape}, reward={reward}")
    while True:
        # Keep car still
        obs, reward, done, infos = env.step([0.0, 0.0])  # stop
        # Check for key press or window close
        key = cv2.waitKey(10)
        if key == ord('q'):
            print("Pressed 'q' to quit.")
            break
