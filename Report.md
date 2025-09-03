# Environment details

The state space has 37 dimensions and consists of:

* The agent's velocity.
* Ray based objects position in the agents forward field of view.

The agent receives a reward of +1 for a yellow banana, and -1 for blue banana. The goal is therefore to maximize the collection of yellow bananas while minimizing / avoiding blue ones.

The action space for the agent consists of the following four possible actions:

0 - walk forward  
1 - walk backward  
2 - turn left  
3 - turn right  
The agent must collect a reward of +13 or more in over 100 consecutive episodes to solve the problem.  

# Learning algorithm
Q-Learning is an approach which generates a Q-table that is used by an agent to determine best action for a given state. This technique becomes difficult and inefficient in environments that have a large state space. Deep Q-Networks on the other hand makes use of a neural network to approximate Q-values for each action based on the input state.

However, there are drawbacks in Deep Q-Learning. A common issue is that the reinforcement learning tends to be unstable or divergent when a non-linear function approximator such as neural networks are used to represent Q. This instability comes from the correlations present in the sequence of observations, the fact that small updates to Q may significantly change the policy and the data distribution, and the correlations between Q and the target values.

To overcome this, experience replay is a technique that was used in this solution that uses the biologically inspired approach of replaying a random sample of prior actions to remove correlations in the observation sequence and smooth changes in the data distribution.

# Model architecture and hyperparameters

Fully connected layer 1: Input 37 (state space), Output 64, RELU activation.  
Fully connected layer 2: Input 64, Output 64, RELU activation.  
Fully connected layer 3: Input 64, Output 4 (action space).  

The hyperparameters for tweaking and optimizing the learning algorithm were:

max_t (3000): maximum number of timesteps per episode.  
eps_start (1.0): starting value of epsilon, for epsilon-greedy action selection.  
eps_end (0.01): minimum value of epsilon.  
eps_decay (0.995): multiplicative factor (per episode) for decreasing epsilon.  

# Plot of rewards
Below is a training run of the above model architecture and hyperparameters:

Number of agents: 1   
Episode 100	Average Score: 4.99
Episode 200	Average Score: 26.14
Episode 221	Average Score: 30.02
Environment solved in 221 episodes!	Average Score: 30.02
The plot of rewards for this run is as follows:
<img width="359" height="234" alt="Trend_solucionDDPGReacher" src="https://github.com/user-attachments/assets/cad2fdb4-7087-4b07-9886-a91773d19ed5" />


# Future work

It might be useful to experiment with other network architectures for this project - different numbers of hidden layers, different numbers of nodes, and additional features such as dropout.

Increasing the size of the experience replay buffer had a major effect on the performance of the agent, and it might perform better with an even larger buffer. It might also be useful to try implementing prioritized experience replay, instead of a random buffer.

The multi-agent learning algorithms mentioned above would also be applicable to this Unity learning application, and it could be informative to explore their effectiveness at this particular task.

Finally, a very simple change to make would be to raise the target score - say, to 40+ - to more accurately gauge the ability of any agent architecture to learn over time.

# References

1.- [DQN Paper] (https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
