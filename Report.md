# Project: Tennis

TBD

## Learning algorithm

For this project, the Deep Deterministic Policy Gradient (DDPG) algorithm was used to train the agent.

Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

This approach is closely connected to Q-learning, and is motivated the same way.

Some characteristics of DDPG:
* DDPG is an off-policy algorithm.
* DDPG can only be used for environments with continuous action spaces.
* DDPG can be thought of as being deep Q-learning for continuous action spaces.

## Model architecture and hyperparameters

The model architectures for the two neural networks used for the Actor and Critic are as follows:

Actor:
* Fully connected layer 1: Input 33 (state space), Output 128, RELU activation, Batch Normalization
* Fully connected layer 2: Input 128, Output 128, RELU activation
* Fully connected layer 3: Input 128, Output 4 (action space), TANH activation

Critic:
* Fully connected layer 1: Input 33 (state space), Output 128, RELU activation, Batch Normalization
* Fully connected layer 2: Input 128, Output 128, RELU activation
* Fully connected layer 3: Input 128, Output 1

## Hyperparameters

```
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay
```

## Plot of rewards

Below is a training run of the above model archicture and hyperparameters:

* Number of agents: 1
* Size of each action: 4
* Environment solved in 103 episodes!	Average Score: 30.21

![Plot of rewards](https://raw.githubusercontent.com/aweeraman/reinforcement-learning-tennis/master/graph.png)

## Ideas for future work

* Training of multiple agents to perform the activity in parallel
* Explore distributed training using
	* A3C - Asynchronous Advantage Actor-Critic
	* A2C - Advantage Actor Critic

## Reference

1 - https://spinningup.openai.com/en/latest/algorithms/ddpg.html
