import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticNet(nn.Module):
    def __init__(self, obs_size=11, h1=128, h2=128):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(obs_size, h1),
            nn.LayerNorm(h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.mu = nn.Linear(h2, 2)
        self.sigma = nn.Linear(h2, 2)
        self.value_head = nn.Linear(h2, 1)

    def forward(self, x):
        features = self.base(x)
        mu = torch.tanh(self.mu(features))
        std = F.softplus(self.sigma(features)) + 1e-3
        value = self.value_head(features)
        return mu, std, value


def select_action(model, state, device):
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    mu, std, value = model(state)
    dist = torch.distributions.Normal(mu, std)
    if model.training:
        action = dist.rsample()
    else:
        action = mu
    log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
    entropy = dist.entropy().sum(dim=-1, keepdim=True)
    action_clipped = torch.clamp(action, -1.0, 1.0)
    return action_clipped.detach().cpu().numpy()[0], log_prob, entropy, value
