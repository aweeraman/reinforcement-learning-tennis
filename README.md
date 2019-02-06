# Project: Tennis Playing Agents

In this environment, two agents control rackets to bounce a ball over a net. If an agent
hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the
ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of
each agent is to keep the ball in play.

![Running agent](https://raw.githubusercontent.com/aweeraman/reinforcement-learning-tennis/master/images/running_agent.png)

The observation space consists of 8 variables corresponding to the position and velocity
of the ball and racket. Each agent receives its own, local observation. Two continuous
actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average
score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

The steps below will describe how to get this running on MacOS:

## 1. Clone the repo

```
$ git clone https://github.com/aweeraman/reinforcement-learning-tennis.git
```

## 2. Install Python & dependencies

Using the Anaconda distribution, create a new python runtime and install the required dependencies:

```
$ conda create -n rl python=3.6
$ source activate rl
$ pip install -r requirements.txt
```

## 3. Install the Unity Environment

Download a pre-built environment to run the agent. You will not need to install Unity for this. The
environment is OS specific, so the correct version for the operating system must be downloaded.

For MacOS, [use this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)

After uncompressing, there should be a directory called "Tennis.app" in the root directory of the repository.

## 4. Train the agent

To train the agent, execute the following:

```
$ python train.py
```

## 5. Run the trained agent

To run the trained agent:

```
$ python tennis.py
```

---

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

![Plot of rewards](https://raw.githubusercontent.com/aweeraman/reinforcement-learning-tennis/master/images/plot_of_rewards.png)

## Future work

* Use a shared critic for the two agents
* Explore distributed training using
	* A3C - Asynchronous Advantage Actor-Critic
	* A2C - Advantage Actor Critic

# Troubleshooting Tips

If you run into an error such as the following when training the agent:

```
ImportError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. See the Python documentation for more information on installing Python as a framework on Mac OS X. Please either reinstall Python as a framework, or try one of the other backends. If you are using (Ana)Conda please install python.app and replace the use of 'python' with 'pythonw'. See 'Working with Matplotlib on OSX' in the Matplotlib FAQ for more information.
```

Modify ~/.matplotlib/matplotlibrc and add the following line:

```
backend: TkAgg
```

## Reference

1. https://spinningup.openai.com/en/latest/algorithms/ddpg.html
2. https://blog.openai.com/learning-to-cooperate-compete-and-communicate/
