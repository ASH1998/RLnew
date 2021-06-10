import gym
import random
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from collections import Counter
import numpy as np
from statistics import median, mean

LR = 1e-3
steps = 500
env = gym.make("CartPole-v0")
env.reset()
score = 50
init_games = 10000

def random_games():
    for epis in range(5):
        env.reset()
        for t in range(steps):
            env.render()
            action = env.action_space.sample()
            obs, rew, done, info = env.step(action)
            print(t, info)
            print(obs)
            print(rew)
            print()
            if done:
                break
random_games()

