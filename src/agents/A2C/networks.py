import torch
import torch.nn as nn


class ActorCriticNet(nn.Module):
    def __init__(self, obs_size=11, h1=128, h2=64, h3=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
        )

        self.mu = nn.Linear(h3, 2)
        self.log_std = nn.Parameter(torch.zeros(1, 2))
        self.value_head = nn.Linear(h3, 1)

    def forward(self, x):
        x = self.shared(x)
        mu = torch.tanh(self.mu(x))
        value = self.value_head(x)
        std = torch.exp(self.log_std)
        return mu, std, value


def select_action(model, state, device):
    state = torch.tensor(state, dtype=torch.float32, device=device)
    mu, std, value = model(state)
    dist = torch.distributions.Normal(mu, std)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
    entropy = dist.entropy().sum(dim=-1, keepdim=True)
    action = torch.clamp(action, -1.0, 1.0)
    return action.cpu().numpy(), log_prob, entropy, value
