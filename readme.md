# RL-based Minesweeper solver
This repository contains code for a Minesweeper solver using Reinforcement Learning. It utilizes Double Deep Q-Learning (DDQN) with Prioritized Experience Replay (PER).  

## Implementation
Minesweeper has been implemented using object oriented programming. Gymnasium is then used to create a gym environment which allows agents to play Minesweeper. The Minesweeper environment is wrapped by a action mask environment which prevents the agent from trying to clear already cleared cells. 

### Minesweeper.py
The Minesweeper.py file contains the implementation of minesweeper.

### Minesweeper_env.py
This file implements the custom Minesweeper environment for Gymnasium, enabling reinforcement learning agents to play Minesweeper. This also includes the action mask which prevents the agent from performing invalid actions.

### DDQN_agent.py
This file contains the implementations of PER, DDQN and the neural network as well as the Action mask policy.
The PER and DDQN are implemented as subclasses of SB3s replay buffer and DQN classes.

## Hyperparameters
Below the different hyperparameters are explained and the chosen values are motivated.

### Alpha

### Gamma

### Learning Rate

### Batch Size

### Exploration Fraction

### Initial Exploration Rate

### Final Exploration Rate

### Number of environments

### Number of timesteps

## Results
At the moment no meaningful results have been produced.