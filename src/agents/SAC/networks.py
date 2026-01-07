import torch
import torch.nn as nn
from torch.distributions import Normal

class QNetwork(nn.Module):
    """Critic (Q) network - approximates the action-value function using a 2-layer MLP."""
    def __init__(self, obs_size, action_dim, h1=256, h2=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size + action_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1)
        )

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)
        return self.net(xu)

class GaussianActor(nn.Module):
    """Stochastic Actor network - outputs Gaussian distribution parameters for actions."""
    def __init__(self, obs_size, action_dim, h1=256, h2=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.base = nn.Sequential(
            nn.Linear(obs_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU()
        )

        self.mu = nn.Linear(h2, action_dim)
        self.log_std = nn.Linear(h2, action_dim)

    def forward(self, state):
        x = self.base(state)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, state):
        """Samples actions using the reparameterization trick and applies Tanh squashing."""
        mu, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mu, std)
        x_t = normal.rsample()
        
        action = torch.tanh(x_t)
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, torch.tanh(mu)

def select_action(model, state, device, deterministic=False):
    """Selects an action from the actor model for environment interaction."""
    state = torch.FloatTensor(state).to(device).unsqueeze(0)
    with torch.no_grad():
        action, _, mu = model.sample(state)
    
    if deterministic:
        return mu.cpu().numpy()[0]
    return action.cpu().numpy()[0]