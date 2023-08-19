import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from QLearning import QLearning

env = gym.make('CartPole-v1')
(state, _) = env.reset()

upper_bounds = env.observation_space.high
lower_bounds = env.observation_space.low
cart_vel_min = -3
cart_vel_max = 3
pole_angle_vel_min = -10
pole_angle_vel_max = 10
upper_bounds[1] = cart_vel_max
lower_bounds[1] = cart_vel_min
upper_bounds[3] = pole_angle_vel_max
lower_bounds[3] = pole_angle_vel_min

bin_size = 30

alpha = 0.1
gamma = 1
epsilon = 0.2
number_of_episodes = 15000

Q1 = QLearning(env, alpha, gamma, epsilon, number_of_episodes, bin_size, lower_bounds, upper_bounds)

Q1.train()

(obtained_rewards_optimal, env1) = Q1.simulate_learned_strategy()

plt.figure(figsize= (12, 5))
plt.plot(Q1.sum_rewards_episode, color = 'blue', linewidth = 1)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.yscale('log')
plt.show()
plt.savefig('convergence.png')

env1.close()

(obtained_rewards_random, env2) = Q1.simulate_random_strategy()
plt.hist(obtained_rewards_random)
plt.xlabel('Sum of rewards')
plt.ylabel('Percentage')
plt.savefig('histogram.png')
plt.show()

(obtained_rewards_optimal, env1) = Q1.simulate_learned_strategy()