# Description of the environment
This environment created by Unity ML is one of the nice playgrounds to test Deep Reinforcement learning algorithms. We are going to solve it using an implementation of DDPG algorithm.

![reacher](https://github.com/user-attachments/assets/dfa415e5-a67d-4056-b630-83e14f0e5d9b)

Animated scene of twenty arms following the target. The solution implemented in this repository is for one arm.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

# How to prepare your system to execute

## Clone the github repository

``` git clone https://github.com/pedrojlucas/DRL_ContinuousControl ```

## Install all the dependencies in a Anaconda or miniconda environment

```
$ conda create -n dqn python=3.6
$ conda activate dqn
$ pip install -r requirements.txt

```

## Install Unity environment

For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

* Linux: [click here]((https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip))
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

Then, place the file in the cloned repository folder in your local machine, and unzip (or decompress) the file.

You will need to change the target folder in the Jupyter notebook file before executing it.
