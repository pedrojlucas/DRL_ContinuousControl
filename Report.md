# Environment details

In this project, we teach an AI reinforcement learning agent in the Unity Reacher environment to direct a double-jointed robot arm to a target location - marked below by a green bubble - and to maintain contact with the target location for as long as possible. A reward of +0.1 is given for each time step that the agent's hand is in the target location, and the environment is considered solved when the robot arm agent attains an average score of 30+ points over 100 consecutive episodes.

The state space in this environment has 33 variables corresponding to the position, rotation, velocity, and angular velocities of the robot arm. The action space is a vector of 4 variables corresponding to the torque applied to the two joints of the robot arm, and each number in the action vector is clipped between -1 and 1. This environment is marked by a continuous action space, with a highly variable range of potential action values and a wide range of motion for the arm to use.

There are two versions of this project environment. The first version contains a single agent; the second version contains 20 identical agents operating in their own copies of the environment, whose learning experience is gathered and then shared across all the agents. For my own implementation, I've chosen to work with the 1 single agent environment. This type of multi-agent learning is useful for AI algorithms like proximal policy optimization (PPO), asynchronous methods, and distributed distributional deterministic policy gradients (DDPG).

# Learning algorithm

The reinforcement learning algorithm being used in this project is deep deterministic policy gradients, or DDPG. DDPG combines the strengths of policy-based (stochastic) and value-based (deterministic) AI learning methods by using two agents, called the Actor and the Critic. The actor directly estimates the optimal policy, or action, for a given state, and applies gradient ascent to maximize rewards. The critic takes the actor's output and uses it to estimate the value (or cumulative future reward) of state-action pairs. The weights of the actor are then updated with the critic’s output, and the critic is updated with the gradients from the temporal-difference error signal at each step. This hybrid algorithm can be a very robust form of artificial intelligence, because it needs fewer training samples than a purely policy-based agent, and demonstrates more stable learning than a purely value-based one.

# Important considerations

In order to achieve an stable and reliable training of the agent some important considerations are taking into account:

* Modifying the Agent.step() method to accommodate multiple agents, and to employ a learning interval. This ensures that the agent performs the learning step only once every 20 time steps during training, and each time 10 passes are made through experience sampling and the Agent.learn() method:

  ``` def step(self, state, action, reward, next_state, done, count):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn every x steps (in this case 10), if enough samples are available in memory, and learn 5 times in that moment.
        if len(self.memory) > BATCH_SIZE:
            if count % 10 == 0: 
                for i in range(1,5): 
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA) ```

* Adding gradient clipping to the critic's loss in the Agent.learn() method. This bounds the upper limits of the gradients close to 1, and prevents the 'exploding gradient problem', in which a network risks making the updated weights too large to properly learn from:

  ``` if GRAD_CLIPPING > 0:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), GRAD_CLIPPING) # added to improve training ```

* In the OUNoise.sample() method, changing random.random() to np.random.randn(). This means that the random noise being added to the experience replay buffer samples via the Ornstein-Uhlenbeck process follows a Gaussian distribution, and turns out to perform much better than a completely random distribution in this case!

  ``` dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))]) ```

# Model architecture and hyperparameters

## Architecture for the Actor

Fully connected layer 1: Input 33 (state space), Output 256, RELU activation.    
Batch normalization layer 1: Input 256 and output 256. Smooth training.  
Fully connected layer 2: Input 256, Output 256, RELU activation.  
Fully connected layer 3: Input 256, Output 4 (action space), tanh activation for clipping output between -1 and 1.   

## Architecture for the Critic

Fully connected layer 1: Input 33 (state space), Output 256, RELU activation.    
Batch normalization layer 1: Input 256 and output 256. Smooth training.  
Fully connected layer 2: Input 256, Output 256, RELU activation.    
Fully connected layer 3: Input 256, Output 4 (action space).     

The hyperparameters for tweaking and optimizing the learning algorithm were:

max_t (3000): maximum number of timesteps per episode.    
BUFFER_SIZE = int(1e6)  replay buffer size  
BATCH_SIZE = 256        minibatch size  
GAMMA = 0.99            discount factor  
TAU = 1e-3              for soft update of target parameters  
LR_ACTOR = 1e-4         learning rate of the actor   
LR_CRITIC = 1e-3        learning rate of the critic  
WEIGHT_DECAY = 0        L2 weight decay  
GRAD_CLIPPING = 1       Activate gradient clippin in critic.  

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

1.- [DDPG Paper] (https://arxiv.org/abs/1509.02971)
