import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.SAC.networks import GaussianActor, QNetwork


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.FloatTensor(np.array(state)),
            torch.FloatTensor(np.array(action)),
            torch.FloatTensor(np.array(reward)).unsqueeze(1),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(np.array(done)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


def soft_update(target, source, tau):
    """Slowly synchronize target network weights using Polyak averaging."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class SACAgent(nn.Module):
    def __init__(self, obs_size, action_dim, device, hidden_dim=256, lr=3e-4):
        super().__init__()
        self.device = device
        self.action_dim = action_dim

        self.actor = GaussianActor(obs_size, action_dim, hidden_dim, hidden_dim).to(
            device
        )

        self.q1 = QNetwork(obs_size, action_dim, hidden_dim, hidden_dim).to(device)
        self.q2 = QNetwork(obs_size, action_dim, hidden_dim, hidden_dim).to(device)

        self.q1_target = QNetwork(obs_size, action_dim, hidden_dim, hidden_dim).to(
            device
        )
        self.q2_target = QNetwork(obs_size, action_dim, hidden_dim, hidden_dim).to(
            device
        )
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )

        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update_parameters(self, memory, batch_size, gamma, tau):
        """Update Actor, Critic and Alpha parameters using a batch of transitions."""
        state, action, reward, next_state, done = memory.sample(batch_size)
        state, action, reward, next_state, done = (
            state.to(self.device),
            action.to(self.device),
            reward.to(self.device),
            next_state.to(self.device),
            done.to(self.device),
        )

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            q1_next = self.q1_target(next_state, next_action)
            q2_next = self.q2_target(next_state, next_action)

            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * gamma * min_q_next

        curr_q1 = self.q1(state, action)
        curr_q2 = self.q2(state, action)

        q_loss = F.mse_loss(curr_q1, target_q) + F.mse_loss(curr_q2, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        new_action, log_prob, _ = self.actor.sample(state)
        q1_new = self.q1(state, new_action)
        q2_new = self.q2(state, new_action)
        min_q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        soft_update(self.q1_target, self.q1, tau)
        soft_update(self.q2_target, self.q2, tau)

        return q_loss.item(), actor_loss.item(), self.alpha.item()
