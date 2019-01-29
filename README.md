# Udacity Deep Reinforcement Learning Nanodegree Project 3: Tennis

TBD

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
