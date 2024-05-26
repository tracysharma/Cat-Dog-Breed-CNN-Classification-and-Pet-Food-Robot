import gymnasium as gym
import panda_gym
import matplotlib.pyplot as plt
import pybullet as p
import numpy as np
import math
from collections import defaultdict
import pickle
import torch
import random
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# env = gym.make("PandaSlide-v3", render_mode="human")

TRAIN = False  # if set to false will skip training, load the last saved model and use that for testing

EPISODES = 1500                 # number of episodes to run the training for
LEARNING_RATE = 0.00025         # the learning rate for optimising the neural network weights.
MEM_SIZE = 10000                # maximum size of the replay memory - will start overwritting values once this is exceed
REPLAY_START_SIZE = 5000       # The amount of samples to fill the replay memory with before we start learning
BATCH_SIZE = 32                 # Number of random samples from the replay memory we use for training each iteration
GAMMA = 0.99                    # Discount factor
EPS_START = 0.2                 # Initial epsilon value for epsilon greedy action sampling
EPS_END = 0.05                # Final epsilon value
EPS_DECAY = 4 * MEM_SIZE        # Amount of samples we decay epsilon over
MEM_RETAIN = 0.1                # Percentage of initial samples in replay memory to keep - for catastrophic forgetting
NETWORK_UPDATE_ITERS = 1000     # Number of samples 'C' for slowly updating the target network \hat{Q}'s weights with the policy network Q's weights

FC1_DIMS = 128                   # Number of neurons in our MLP's first hidden layer
FC2_DIMS = 128                   # Number of neurons in our MLP's second hidden layer

# metrics for displaying training status
best_reward = 0
average_reward = 0
episode_history = []
episode_reward_history = []
np.bool = np.bool_

# for creating the policy and target networks - same architecture
class Network(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        full_obs = env.unwrapped.get_obs()
        self.input_shape = full_obs['observation'].shape
        self.action_space = env.action_space.n

        print("Input shape: ", self.input_shape)
        print("Space shape: ", self.action_space)

        # build an MLP with 2 hidden layers
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(*self.input_shape, FC1_DIMS),   # input layer
            torch.nn.ReLU(),     # this is called an activation function
            torch.nn.Linear(FC1_DIMS, FC2_DIMS),    # hidden layer
            torch.nn.ReLU(),     # this is called an activation function
            torch.nn.Linear(FC2_DIMS, self.action_space)    # output layer
            )

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()  # loss function

    def forward(self, x):
        return self.layers(x)
    
# handles the storing and retrival of sampled experiences
class ReplayBuffer:
    def __init__(self, env):
        self.mem_count = 0
        full_obs = env.unwrapped.get_obs()
        print("Full obs: ", full_obs)
        self.states = np.zeros((MEM_SIZE, *full_obs['observation'].shape),dtype=np.float32)
        # self.states = np.zeros((MEM_SIZE, *env.robot.get_ee_position().shape),dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, *full_obs['observation'].shape),dtype=np.float32)
        # self.states = np.zeros((MEM_SIZE, *env.robot.get_ee_position().shape),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool)

    def add(self, state, action, reward, state_, done):
        # if memory count is higher than the max memory size then overwrite previous values
        if self.mem_count < MEM_SIZE:
            mem_index = self.mem_count
        else:
            ############ avoid catastropic forgetting - retain initial 10% of the replay buffer ##############
            mem_index = int(self.mem_count % ((1-MEM_RETAIN) * MEM_SIZE) + (MEM_RETAIN * MEM_SIZE))
        
        # print("Action: ", action)
        # print("State: ", state['observation'])
        # print("State_: ", state_)

        self.states[mem_index]  = state['observation']
        # self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_['observation']
        # self.states_[mem_index] = state_
 
        # checking values
        # print("states: ", self.states[mem_index])
        # print("actions: ", self.actions[mem_index])
        # print("rewards: ", self.rewards[mem_index])
        # print("states_: ", self.states_[mem_index])
        # print("dones: ", self.dones[mem_index])
        # print(" ")

        self.mem_count += 1

    # returns random samples from the replay buffer, number is equal to BATCH_SIZE
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)

        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones

class DQN_Solver:
    def __init__(self, env):
        self.memory = ReplayBuffer(env)
        self.policy_network = Network(env)  # Q
        self.target_network = Network(env)  # \hat{Q}
        self.target_network.load_state_dict(self.policy_network.state_dict())  # initially set weights of Q to \hat{Q}
        self.learn_count = 0    # keep track of the number of iterations we have learnt for

    # epsilon greedy
    def choose_action(self, observation):
        # only start decaying epsilon once we actually start learning
        if self.memory.mem_count > REPLAY_START_SIZE:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * self.learn_count / EPS_DECAY)
        else:
            eps_threshold = 1.0
        # if we rolled a value lower than epsilon sample a random action
        if random.random() < eps_threshold:
            return np.random.choice(self.policy_network.action_space, p=[1/5, 1/5, 1/5, 1/5, 1/10, 1/10])

        # otherwise policy network, Q, chooses action with highest estimated Q-value so far
        state = torch.tensor(observation).float().detach()
        state = state.unsqueeze(0)
        # print("STATE: ", state)
        self.policy_network.eval()  # only need forward pass
        with torch.no_grad():       # so we don't compute gradients - save memory and computation
            q_values = self.policy_network(state)
        return torch.argmax(q_values).item()
    

    # main training loop
    def learn(self):
        states, actions, rewards, states_, dones = self.memory.sample()  # retrieve random batch of samples from replay memory
        states = torch.tensor(states , dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        states_ = torch.tensor(states_, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        self.policy_network.train(True)
        q_values = self.policy_network(states)                # get current q-value estimates (all actions) from policy network, Q
        q_values = q_values[batch_indices, actions]           # q values for sampled actions only

        self.target_network.eval()                            # only need forward pass
        with torch.no_grad():                                 # so we don't compute gradients - save memory and computation
            ###### get q-values of states_ from target network, \hat{q}, for computation of the target q-values ###############
            q_values_next = self.target_network(states_)
            ###################################################################################################################

        q_values_next_max = torch.max(q_values_next, dim=1)[0]  # max q values for next state

        q_target = rewards + GAMMA * q_values_next_max * dones  # our target q-value

        ###### compute loss between target (from target network, \hat{Q}) and estimated q-values (from policy network, Q) #########
        loss = self.policy_network.loss(q_target , q_values)
        ###########################################################################################################################

        #compute gradients and update policy network Q weights
        self.policy_network.optimizer.zero_grad()
        loss.backward()
        self.policy_network.optimizer.step()
        self.learn_count += 1

        # set target network \hat{Q}'s weights to policy network Q's weights every C steps
        if  self.learn_count % NETWORK_UPDATE_ITERS == NETWORK_UPDATE_ITERS - 1:
            print("updating target network")
            self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def returning_epsilon(self):
        return self.exploration_rate

############################################################################################
# Train network
if TRAIN:
    env = gym.make("PandaSlide-v3")
    # env = gym.make("PandaSlideDense-v3")
    # set manual seeds so we get same behaviour everytime - so that when you change your hyper parameters you can attribute the effect to those changes
    env.action_space.seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    episode_batch_score = 0
    episode_reward = 0
    agent = DQN_Solver(env)  # create DQN agent
    plt.clf()

    for i in range(EPISODES):
        state, info = env.reset()  # this needs to be called once at the start before sending any actions
        step_counter = 0
        # print("EPISODE: ", i)
        while True:
            # print("Memcount: ", agent.memory.mem_count)
            # sampling loop - sample random actions and add them to the replay buffer
            # print(state)
            action = agent.choose_action(state['observation'])
            # print(action, end=" ")
            state_, reward, done, _, info = env.step(action)

            ####### add sampled experience to replay buffer ##########
            agent.memory.add(state, action, reward, state_, done)
            ##########################################################

            # print(agent.memory.mem_count, " : ", i)
            # only start learning once replay memory reaches REPLAY_START_SIZE
            if agent.memory.mem_count > REPLAY_START_SIZE:
                agent.learn()

            # Exit when steps over this amount
            step_counter += 1
            # print(step_counter) 
            if step_counter >= 40:
                done = True

            state = state_
            episode_batch_score += reward
            episode_reward += reward

            if done:
                break

        episode_history.append(i)
        episode_reward_history.append(episode_reward)
        episode_reward = 0.0

        # save our model every batches of 100 episodes so we can load later. (note: you can interrupt the training any time and load the latest saved model when testing)
        if i % 100 == 0 and agent.memory.mem_count > REPLAY_START_SIZE:
            torch.save(agent.policy_network.state_dict(), "policy_network.pkl")
            print("average total reward per episode batch since episode ", i, ": ", episode_batch_score/ float(100))
            episode_batch_score = 0
        elif agent.memory.mem_count < REPLAY_START_SIZE:
            print("waiting for buffer to fill...")
            episode_batch_score = 0

    plt.plot(episode_history, episode_reward_history)
    plt.show()

############################################################################################

# Test trained policy
import os
print("Current Working Directory:", os.getcwd())

# env = gym.make("PandaSlideDense-v3", render_mode="human")
env = gym.make("PandaSlide-v3", render_mode = "human")
agent = DQN_Solver(env)
agent.policy_network.load_state_dict(torch.load("policy_network.pkl"))
# state = env.reset()
agent.policy_network.eval()

state, info = env.reset()

for _ in range(100):
    with torch.no_grad():
        # q_values = agent.policy_network(torch.tensor(state['observation'], dtype=torch.float32))
        # state['observation'] contains, xyz position and vel, goal position
        q_values = agent.policy_network(torch.tensor(state['observation'], dtype=torch.float32))
        print("Q Values: ", q_values)
    action = torch.argmax(q_values).item() # select action with highest predicted q-value
    state, reward, terminated, _, info = env.step(action)

    print("State: ", state['observation'])
    print("Action: ", action)
    print("Reward: ", reward)
    print(" ")
    # time.sleep(10)

    # if done:
    #     break

env.close()
