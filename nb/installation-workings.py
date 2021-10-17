#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


import numpy as np


# In[ ]:


get_ipython().system('pip install gym')


# In[ ]:


# !pip install gym[all]


# In[1]:


import gym 
env = gym.make('CartPole-v1')

# env is created, now we can use it: 
for episode in range(10): 
    obs = env.reset()
    for step in range(200):
        action = env.action_space.sample()  # or given a custom model, action = policy(observation)
        nobs, reward, done, info = env.step(action)


# In[ ]:


nobs


# In[ ]:


reward


# In[ ]:


get_ipython().system('pip install gym[atari]')


# In[ ]:


get_ipython().system('pip install stable-baselines3[extra]')


# In[ ]:





# In[2]:


import gym 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os


# In[3]:


environment_name = "CartPole-v0"


# In[4]:


episodes = 50
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()


# In[ ]:


env.action_space.sample()


# In[ ]:


env.observation_space.sample()


# In[ ]:


env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose = 1)


# In[ ]:


model.learn(total_timesteps=20000)


# In[ ]:


PPO_path = os.path.join('Training', 'Saved Models', 'PPO_model')


# In[ ]:


model.save(PPO_path)


# In[ ]:


# del model
model = PPO.load(PPO_path, env=env)


# In[ ]:


from stable_baselines3.common.evaluation import evaluate_policy


# In[ ]:


evaluate_policy(model, env, n_eval_episodes=50, render=True)


# In[ ]:


env.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




