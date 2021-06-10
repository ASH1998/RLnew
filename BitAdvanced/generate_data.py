import gym
import random
import numpy as np
from statistics import mean, median
from collections import Counter
from tqdm import trange

env = gym.make("CartPole-v0")
goal_steps = 500
score_requirement = 50
initial_games = 50000

def init_pop():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in trange(initial_games):
        score = 0
        game = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            #env.reset()
            action = random.randrange(0,2)
            obs, rew, done, info = env.step(action)

            if len(prev_obs)>0:
                game.append([prev_obs, action])
            prev_obs = obs
            score+=rew
            if done:
                break

        if score>=score_requirement:
            accepted_scores.append(score)
            for data in game:
                if data[1]==1:
                    output = [0,1]
                elif data[1] ==0:
                    output = [1,0]

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save("saved.npy", training_data_save)

    print("AVG accepted score : ", mean(accepted_scores))
    print("Median score  : ", median(accepted_scores))
    print(Counter(accepted_scores))

init_pop()
