import gymnasium as gym
import matplotlib.pyplot as plt
import torch

from QLearning import Q_Learning


env = gym.make('CartPole-v1')

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Q1 = Q_Learning(env, device, 128, 0.99, 0.9, 0.05, 1000, 0.005, 1e-4)

Q1.train(150)
Q1.plot_durations(True)

env1 = gym.make('CartPole-v1')

Q1.simulate_learned_strategy(env1)