# Project: Tennis

In this environment, two agents control rackets to bounce a ball over a net. If an agent
hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the
ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of
each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity
of the ball and racket. Each agent receives its own, local observation. Two continuous
actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average
score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

## Learning algorithm

For this project, the Multi-Agent Deep Deterministic Policy Gradient (DDPG) algorithm was used to train the agent.

Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

This approach is closely connected to Q-learning, and is motivated the same way.

Some characteristics of DDPG:
* DDPG is an off-policy algorithm.
* DDPG can only be used for environments with continuous action spaces.
* DDPG can be thought of as being deep Q-learning for continuous action spaces.

This approach extends DDPG to support multiple agents, where both agents are able to access
the observations and actions of each other and cooperate to keep the ball in play for as long as
possible.

## Model architecture and hyperparameters

The model architectures for the two neural networks used for the Actor and Critic are as follows:

Actor:
* Fully connected layer 1: Input 48 (state space), Output 512, RELU activation, Batch Normalization
* Fully connected layer 2: Input 512, Output 256, RELU activation
* Fully connected layer 3: Input 256, Output 2 (action space), TANH activation

Critic:
* Fully connected layer 1: Input 48 (state space), Output 512, RELU activation, Batch Normalization
* Fully connected layer 2: Input 512, Output 256, RELU activation
* Fully connected layer 3: Input 256, Output 1

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

* Number of agents: 2
* Size of each action: 2
* Environment solved in 2314 episodes!	Average Score: 0.50

![Plot of rewards](https://raw.githubusercontent.com/aweeraman/reinforcement-learning-tennis/master/graph.png)

## Ideas for future work

* Use a shared critic for the two agents
* Explore distributed training using
	* A3C - Asynchronous Advantage Actor-Critic
	* A2C - Advantage Actor Critic

## Reference

1 - https://spinningup.openai.com/en/latest/algorithms/ddpg.html
2 - https://blog.openai.com/learning-to-cooperate-compete-and-communicate/
