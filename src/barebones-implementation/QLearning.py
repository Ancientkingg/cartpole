import numpy as np
import gymnasium as gym
import time

class QLearning:
    
    def __init__(self, env, alpha, gamma, epsilon, episode_size, bin_size, lower_bounds, upper_bounds):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_number = env.action_space.n
        self.episode_size = episode_size
        self.bin_size = bin_size
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        self.sum_rewards_episode = []

        self.q_table = np.random.uniform(low = 0, high = 1, size = ([bin_size] * len(lower_bounds) + [self.action_number]))


    def simulate_random_strategy(self):
        env2 = gym.make('CartPole-v1')
        (current_state, _) = env2.reset()

        episodes = 100
        time_steps = 1000

        sum_rewards_episodes = []

        for episode_idx in range(episodes):
            rewards_current_episode = []
            initial_state = env2.reset();
            # print(episode_idx)

            for time_idx in range(time_steps):
                random_action = env2.action_space.sample()
                observation, reward, terminated, truncated, info = env2.step(random_action)
                rewards_current_episode.append(reward)

                if terminated or truncated:
                    break
                sum_rewards_episodes.append(np.sum(rewards_current_episode))

        return sum_rewards_episodes, env2
    
    def discretize(self, state):
        position = state[0]
        velocity = state[1]
        angle = state[2]
        angular_velocity = state[3]

        cart_pos_bin = np.linspace(self.lower_bounds[0], self.upper_bounds[0], self.bin_size)
        cart_vel_bin = np.linspace(self.lower_bounds[1], self.upper_bounds[1], self.bin_size)
        pole_angle_bin = np.linspace(self.lower_bounds[2], self.upper_bounds[2], self.bin_size)
        pole_vel_bin = np.linspace(self.lower_bounds[3], self.upper_bounds[3], self.bin_size)

        index_pos = np.maximum(np.digitize(position, cart_pos_bin) - 1, 0)
        index_vel = np.maximum(np.digitize(velocity, cart_vel_bin) - 1, 0)
        index_angle = np.maximum(np.digitize(angle, pole_angle_bin) - 1, 0)
        index_vel_angle = np.maximum(np.digitize(angular_velocity, pole_vel_bin) - 1, 0)

        return tuple([index_pos, index_vel, index_angle, index_vel_angle])

    def determine_action(self, state, index):
        
        if (index < 500):
            return np.random.choice(self.action_number)
        
        random_number = np.random.random()

        if index > 7000:
            self.epsilon = 0.999 * self.epsilon
        
        if random_number < self.epsilon:
            return np.random.choice(self.action_number)
        else:
            return np.random.choice(np.where(
                self.q_table[self.discretize(state)] == 
                             np.max(self.q_table[self.discretize(state)]))[0])
        
    def train(self):
        for episode_idx in range(self.episode_size):
            rewards_episode = []

            (state_S, _) = self.env.reset()
            state_S = list(state_S)

            if (episode_idx % 500 == 0):
                print("Simulating episode number {}".format(episode_idx))

            terminal_state = False

            while not terminal_state:
                state_S_idx = self.discretize(state_S)

                action_A = self.determine_action(state_S, episode_idx)

                (state_S_prime, reward_R, terminal_state, _, _) = self.env.step(action_A)

                rewards_episode.append(reward_R)

                state_S_prime = list(state_S_prime)

                state_S_prime_idx = self.discretize(state_S_prime)

                q_max_prime = np.max(self.q_table[state_S_prime_idx])

                if not terminal_state:
                    error = reward_R + self.gamma * q_max_prime - self.q_table[state_S_idx + (action_A, )]
                    self.q_table[state_S_idx + (action_A, )] = self.q_table[state_S_idx + (action_A, )] + self.alpha * error
                else:
                    error = reward_R - self.q_table[state_S_idx + (action_A, )]
                    self.q_table[state_S_idx + (action_A, )] = self.q_table[state_S_idx + (action_A, )] + self.alpha * error

                state_S = state_S_prime

                # print("Sum of rewards in episode {}: {}".format(episode_idx, np.sum(rewards_episode)))
                self.sum_rewards_episode.append(np.sum(rewards_episode))

    def simulate_learned_strategy(self):
        env1 = gym.make('CartPole-v1', render_mode = 'human')
        (current_state, _) = env1.reset()

        time_steps = 50000
        obtained_rewards = []

        for time_idx in range(time_steps):
            # print(time_idx)
            action_in_state_S = np.random.choice(np.where(self.q_table[self.discretize(current_state)]==np.max(self.q_table[self.discretize(current_state)]))[0])
            
            current_state, reward, terminated, truncated, info = env1.step(action_in_state_S)

            obtained_rewards.append(reward)

            time.sleep(0.02)
            if (terminated or truncated):
                print("Obtained rewards: {}".format(np.sum(obtained_rewards)))
                print("Time steps: {}".format(time_idx))
                time.sleep(1)
                break

        return obtained_rewards, env1
