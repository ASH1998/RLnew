import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

winratepercnt = []
scores = []

env = gym.make('FrozenLake8x8-v0')


singlespace = spaces.Discrete(4)

nbatches = 1000

for i in trange(nbatches):
    env.reset()
    done = False
    score = 0

    while not done:
        action = singlespace.sample()
        obs, reward, done, info = env.step(action)
        score += reward
    
    scores.append(score)
    if i%100==0:
        average = np.mean(scores[-100:])
        winratepercnt.append(average)

plt.plot(winratepercnt)
plt.show()

