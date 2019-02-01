from unityagents import UnityEnvironment
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from ddpg_agent import Agent, ReplayBuffer
from collections import deque

env = UnityEnvironment(file_name='Tennis.app')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

state_size = state_size * 2
agent_1 = Agent(state_size=state_size, action_size=action_size, random_seed=1)
agent_2 = Agent(state_size=state_size, action_size=action_size, random_seed=1)

actor_1_weights = "actor_1_model.pth"
actor_2_weights = "actor_2_model.pth"

critic_1_weights = "critic_1_model.pth"
critic_2_weights = "critic_2_model.pth"

def ddpg(n_episodes=100000, max_t=20000):

    scores_deque = deque(maxlen=100)
    total_scores = []
    average_scores = []

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment
        states = env_info.vector_observations                  # get the current state (for each agent)
        states = np.reshape(states, (1, state_size))
        agent_1.reset()
        agent_2.reset()
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)

        while True:
            action_1 = agent_1.act(states, add_noise=True)
            action_2 = agent_2.act(states, add_noise=True)
            actions = np.concatenate((action_1, action_2), axis=0)
            actions = np.reshape(actions, (1, 4))

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations         # get next state (for each agent)
            next_states = np.reshape(next_states, (1, state_size))
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished

            agent_1.step(states, action_1, rewards[0], next_states, dones[0])
            agent_2.step(states, action_2, rewards[1], next_states, dones[1])

            scores += rewards                                  # update the score (for each agent)
            states = next_states                               # roll over states to next time step

            if np.any(dones):                                  # exit loop if episode finished
                break

        scores_deque.append(np.max(scores))
        total_scores.append(np.max(scores))
        average_scores.append(np.mean(scores_deque))

        torch.save(agent_1.actor_local.state_dict(), actor_1_weights)
        torch.save(agent_2.actor_local.state_dict(), actor_2_weights)
        torch.save(agent_1.critic_local.state_dict(), critic_1_weights)
        torch.save(agent_2.critic_local.state_dict(), critic_2_weights)
        
        print('\rEpisode: \t{} \tScore: \t{:.2f} \tAverage Score: \t{:.2f}'.format(i_episode, np.max(scores), np.mean(scores_deque)), end="")

        if np.mean(scores_deque)>=0.5:  # consider done when the average score reaches 0.5 or more
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            break

    plt.plot(np.arange(1, len(average_scores)+1), average_scores)
    plt.ylabel('Avg Score')
    plt.xlabel('Episode #')
    plt.show()

ddpg()
