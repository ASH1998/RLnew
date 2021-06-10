from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression
import numpy as np
import tflearn
import gym
import random

env = gym.make("CartPole-v0")
score_requirement = 50
goal_steps = 500

train_data = np.load('saved.npy')

def network(input_size):
    net = input_data(shape=[None, input_size, 1], name='input')
    net = fully_connected(net, 128, activation='relu')
    net = dropout(net, 0.8)

    net = fully_connected(net, 256, activation='relu')
    net = dropout(net, 0.8)

    net = fully_connected(net, 512, activation='relu')
    net = dropout(net, 0.8)

    net = fully_connected(net, 256, activation='relu')
    net = dropout(net, 0.8)

    net = fully_connected(net, 128, activation='relu')
    net = dropout(net, 0.8)

    net = fully_connected(net, 2, activation='softmax')
    net = regression(net)

    model = tflearn.DNN(net, tensorboard_dir='E:\\Python Coding\\RL\\log')

    return model

def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(train_data), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = network(input_size=len(X[0]))

    model.fit(X, y, n_epoch=5, show_metric=True)
    return model

model = train_model(train_data)

scores = []
choices = []
for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()

        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score+=reward
        if done: break

    scores.append(score)

print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
print(score_requirement)
