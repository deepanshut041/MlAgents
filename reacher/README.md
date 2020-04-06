# Unity Environment: Push Block

![Banner](../media/push.png)

## Index

- [Environment](#environment)
- [Algorithm](#algorithm)
- [Getting Started](#getting-started)
- [Training](#training)
- [Watch trained Agent](#watch-trained-agent)
- [Reference](#reference)

## Environment

A platforming The Environment where the agent can push a block around. Here the goal is agent must push the block to the goal.

### Reward Function

- -0.0025 for every step.
- +1.0 if the block touches the goal.

### Behavior Parameters

- **Vector Observation space:** (Continuous) 70 variables corresponding to 14 ray-casts each detecting one of three possible objects (wall, goal, or block).
- **Vector Action space:** (Discrete) Size of 6, corresponding to turn clockwise and counterclockwise and move along four different face directions.
- **Visual Observations:**  (Optional): One first-person camera. Use **VisualPushBlock** scene. The visual observation version of this environment does not train with the provided default training parameters.

## Algorithm

We'll be using **DDQN** to solve the environment.

### DDQN

DDQN or Dueling Deep Q Networks is a reinforcement learning algorithm that tries to create a Q value via two function estimators: one that estimates the advantage function, and another that estimates the value function. The value function calculates the value of a given input state, and the advantage function calculates the benefits of taking a given action. Together, these can provide a good estimation of how good the next frame performs given a certain state-action pair. The dual network structure shows better results than single network structures such as DDQN, as each function estimator can focus on a different part of the image, using different strategies to create a better estimator for the Q function.

### Replay Buffer

The inclusion of a replay memory that stores experiences, e<sub>t</sub> = (s<sub>t</sub>, a<sub>t</sub>, r<sub>t</sub>, s<sub>t+1</sub>), at each time step, t, in a data set D<sub>t</sub> = {e<sub>1</sub>, e<sub>2</sub>, … e<sub>t</sub>}. During learning, mini-batches of experiences, U(D), are randomly sampled to update (i.e., train) the Q-network. The random sampling of the pooled set of experiences removes correlations between observed experiences, thereby smoothing over changes in the distribution of experiences to avoid the agent getting stuck in local minimum or oscillating or diverging from the optimal policy.

### Target Network

This target network is only updated periodically, in contrast to the action-value Q-network that is updated at each time step. We'll be not be updating it periodically instead we'll be doing a soft update on the target network. It provides stability to the target network.

### DDPG with Experience Replay

![DDPG Algorithm](./media/ddpg_algorithm.png)

The algorithm is the same as DDQN the only difference is in the neural network.

## Getting Started

### Files

- `ddqn_agent.py`: Implementation of a DDQN-Agent. [Link](https://github.com/deepanshut041/ml_agents-pytorch/blob/master/push-block/dqn_agent.py)
- `replay_memory.py`: Implementation of a DDQN-Agent's replay buffer (memory). [Link](https://github.com/deepanshut041/ml_agents-pytorch/blob/master/push-block/replay_memory.py)
- `model.py`: Implementation of neural network for vector-based DDQN learning using PyTorch. [Link](https://github.com/deepanshut041/ml_agents-pytorch/blob/master/push-block/model.py)
- `env_test.ipynb`: Test if the environment is properly set up or not. [Link](https://github.com/deepanshut041/ml_agents-pytorch/blob/master/push-block/env_test.ipynb)
- `train.ipynb`: Train DDQN Agent on the environment. [Link](https://github.com/deepanshut041/ml_agents-pytorch/blob/master/push-block/train.ipynb)
- `test.ipynb`: Test DDQN-agent using a trained model and visualize it. [Link](https://github.com/deepanshut041/ml_agents-pytorch/blob/master/push-block/test.ipynb)

### How to Train

- **Step 1**: First, get started with `env_test.ipynb` to verify that the environment is properly set up.
- **Step 2**: Train DDQN Agent using `train.ipynb`.
- **Step 3**: Test DDQN-Agent and visualize using `test.ipynb` at different checkpoints. You can also skip **Step 2** and download the pre-trained model from [**here**](https://drive.google.com/open?id=1qn_Nq-uZCHTdsQvs7XgjCgAP8v390Cbb) and place it inside the push-block folder.

## Training

### Neural Network

Because the agent learnings from vector data (not pixel data), the Q-Network (local and target network) employed here consisted of just 2 hidden, fully connected layers with 64 nodes. The size of the input layer was equal to the dimension of the state size (i.e., 20 nodes) and the size of the output layer was equal to the dimension of the action size (i.e., 3).  

### Hyperparameters

- STATE_SIZE: 210 (Dimension of each state)
- ACTION_SIZE: 7 (Two possible actions (Move left, move right))
- BUFFER_SIZE: 1e5 (Replay buffer size)
- BATCH_SIZE: 256 (Mini-batch size)
- GAMMA: 0.99 (Discount factor)
- LR: 1e-3 (Learning rate)
- Tau: 1e-2 (Soft-parameter update)
- UPDATE_EVERY: 5 (How often to update the network)

### Training Parameters

- BENCHMARK_REWARD = 4.3
- epsilon (float): 1.0
- epsilon_min: 0.01
- epsilon_decay: 200
- SCORES_AVERAGE_WINDOW = 100
- NUM_EPISODES = 2000
- MAX_TIMESTEP = 100

### Performance

| Training | Testing |
|:-:|:-:|
| ![training](./media/train.svg) |  ![Testing](./media/test.svg) |

## Watch Trained Agent

![Trained Agent](./media/trained_agent.gif)

## Reference

- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Petersen, S. (2015). Human-Level control through deep reinforcement learning. Nature, 518(7540), 529.
