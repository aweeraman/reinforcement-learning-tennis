from unityagents import UnityEnvironment
import numpy as np
import random
import torch
from ddpg_agent import Agent

env = UnityEnvironment(file_name='Tennis.app')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

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

agent_1.actor_local.load_state_dict(torch.load('actor_1_model.pth'))
agent_1.critic_local.load_state_dict(torch.load('critic_1_model.pth'))

agent_2.actor_local.load_state_dict(torch.load('actor_2_model.pth'))
agent_2.critic_local.load_state_dict(torch.load('critic_2_model.pth'))

for i_episode in range(1, 10):
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    states = np.reshape(states, (1, state_size))
    agent_1.reset()
    agent_2.reset()

    while True:
        action_1 = agent_1.act(states, add_noise=False)
        action_2 = agent_2.act(states, add_noise=False)
        actions = np.concatenate((action_1, action_2), axis=0)
        actions = np.reshape(actions, (1, 4))

        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations         # get next state (for each agent)
        next_states = np.reshape(next_states, (1, state_size))
        dones = env_info.local_done                        # see if episode finished

        states = next_states                               # roll over states to next time step

        if np.any(dones):                                  # exit loop if episode finished
            break
