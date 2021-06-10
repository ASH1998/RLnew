# import gym
# env = gym.make('FrozenLake8x8-v0')
# env.reset()
# for _ in range(10):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()




# from gym import envs
# import gym

# frozen = gym.make('FrozenLake8x8-v0')

# numEpisodes = 10

# for episode in range(numEpisodes):
#     observation = frozen.reset()

#     for epochs in range(100):
#         # frozen.render()
#         # print(observation)
#         action = frozen.action_space.sample()
#         # print(action)

#         obs, reward, done, info = frozen.step(action)
        
#         if done: #this is needed to be changed
#             print("Episode finished after {} timesteps".format(epochs+1))
#             break

# frozen.close()
        
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

ngames = 1000
percentage = []
scores = []

env = gym.make('FrozenLake8x8-v0')

for i in trange(ngames):
    done = False
    obs = env.reset()

    score = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action=action)
        score+=reward
    scores.append(score)

    if i%10==0:
        average = np.mean(scores[-10:])
        percentage.append(average)

plt.plot(percentage)
plt.show()
