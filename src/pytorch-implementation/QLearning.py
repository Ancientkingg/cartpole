
import torch
import torch.optim as optim
import torch.nn as nn
import random
import math
import time
import matplotlib.pyplot as plt
from itertools import count


from DQN import DQN
from ReplayMemory import ReplayMemory
from ReplayMemory import Transition

class Q_Learning:

    def __init__(self, env, device, batch_size, gamma, eps_start, eps_end, eps_decay, tau, learning_rate):
        self.env = env
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.learning_rate = learning_rate
        self.episode_durations = []

        n_actions = env.action_space.n

        state, _ = env.reset()

        n_observations = len(state)

        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = learning_rate, amsgrad = True)

        self.memory = ReplayMemory(10000)

        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)

        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device = self.device, dtype = torch.long)

    def plot_durations(self, show_result = False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype = torch.float)

        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())

        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)

    def train_single_step(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device = self.device, dtype = torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device = self.device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self, num_episodes):

        for episode_idx in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype = torch.float32, device = self.device).unsqueeze(0)

            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device = self.device)

                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype = torch.float32, device = self.device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)

                state = next_state

                self.train_single_step()

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)

                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break

        print('Completed training')
        self.plot_durations(True)
        plt.ioff()
        plt.show()

    def simulate_learned_strategy(self, env):
        current_state, _ = env.reset()
        current_state = torch.tensor(current_state, dtype = torch.float32, device = self.device).unsqueeze(0)

        for time_idx in count():
            # print(time_idx)
            action = self.select_action(current_state)
            
            current_state, reward, terminated, truncated, _ = env.step(action.item())

            time.sleep(0.02)
            if (terminated or truncated):
                print("Time steps: {}".format(time_idx))
                time.sleep(1)
                break

        return env
                    


