import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from replay_memory import ReplayBuffer
from model import Actor, Critic
from ou_noise import OUNoise

class DDPGAgent():
    def __init__(self, input_shape, action_size, buffer_size, batch_size, gamma, lr_actor, lr_critic, tau, update_every, num_agents, device):
        """Initialize an Agent object.
        
        Params
        ======
            input_shape (tuple): dimension of each state
            action_size (int): dimension of each action
            buffer_size (int): replay buffer size
            batch_size (int):  minibatch size
            gamma (float): discount factor
            lr_actor (float): Actor learning rate
            lr_critic (float): Critic learning rate 
            tau (float): Soft-parameter update
            update_every (int): how often to update the network
            num_agents: Number of agent
            device(string): Use Gpu or CPU
        """
        self.input_shape = input_shape
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.update_every = update_every
        self.tau = tau
        self.num_agents = num_agents
        self.device = device

        
        # Actor Network
        self.actor_local = Actor(input_shape, action_size).to(self.device)
        self.actor_target = Actor(input_shape, action_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network
        self.critic_local = Critic(input_shape, action_size).to(self.device)
        self.critic_target = Critic(input_shape, action_size).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic)

        # Noise process
        self.noise = [
            OUNoise(action_size)
            for _ in range(num_agents)
            ]
        
        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.device)
        
         # Make sure target is with the same weight as the source
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

        self.t_step = 0

    
    def step(self, states, actions, rewards, next_states, dones):

        for i in range(0, len(states)):

            # Save experience in replay memory
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
                
    def act(self, states, add_noise=True):
        actions = []
        state = torch.from_numpy(states).to(self.device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        for i in range(0, self.num_agents):
            if add_noise:
                actions.append(np.clip(action[i] + self.noise[i].sample(), -1, 1))
            else:
                actions.append(np.clip(action[i], -1, 1))

        return actions

    def reset(self):
        for i in range(self.num_agents):
            self.noise[i].reset()
        
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    
    # θ'=θ×τ+θ'×(1−τ)
    def soft_update(self, policy_model, target_model, tau):
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau*policy_param.data + (1.0-tau)*target_param.data)
    
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def load_model(self, path):
        
        checkpoint = torch.load(path)
        self.actor_local.load_state_dict(checkpoint['actor_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])

        self.critic_local.load_state_dict(checkpoint['critic_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        scores = checkpoint['scores']

        return scores

    def save_model(self, path, scores):
        model = {
            "actor_dict": self.actor_local.state_dict(),
            "critic_dict": self.critic_local.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "scores": scores
        }
        torch.save(model, path)