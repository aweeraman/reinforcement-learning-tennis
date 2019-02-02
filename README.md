# Udacity Deep Reinforcement Learning Nanodegree Project 3: Tennis

In this environment, two agents control rackets to bounce a ball over a net. If an agent
hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the
ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of
each agent is to keep the ball in play.

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
$ python train.py --run
Mono path[0] = '/Users/anuradha/ninsei/udacity/reinforcement-learning-tennis/Tennis.app/Contents/Resources/Data/Managed'
Mono config path = '/Users/anuradha/ninsei/udacity/reinforcement-learning-tennis/Tennis.app/Contents/MonoBleedingEdge/etc'
INFO:unityagents:
'Academy' started successfully!
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :

Unity brain name: TennisBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 8
        Number of stacked Vector Observation: 3
        Vector Action space type: continuous
        Vector Action space size (per agent): 2
        Vector Action descriptions: ,
Number of agents: 2
Size of each action: 2
There are 2 agents. Each observes a state with length: 24
The state for the first agent looks like: [ 0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.         -6.65278625 -1.5
 -0.          0.          6.83172083  6.         -0.          0.        ]
Episode: 	2414 	Score: 	1.50 	Average Score: 	0.50
Environment solved in 2314 episodes!	Average Score: 0.50
```

![Plot of rewards](https://raw.githubusercontent.com/aweeraman/reinforcement-learning-tennis/master/graph.png)

## 5. Run the trained agent

To run the trained agent:

```
$ python tennis.py
```

![Running agent](https://raw.githubusercontent.com/aweeraman/reinforcement-learning-tennis/master/running_agent.png)

# Troubleshooting Tips

If you run into an error such as the following when training the agent:

```
ImportError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. See the Python documentation for more information on installing Python as a framework on Mac OS X. Please either reinstall Python as a framework, or try one of the other backends. If you are using (Ana)Conda please install python.app and replace the use of 'python' with 'pythonw'. See 'Working with Matplotlib on OSX' in the Matplotlib FAQ for more information.
```

Modify ~/.matplotlib/matplotlibrc and add the following line:

```
backend: TkAgg
```
