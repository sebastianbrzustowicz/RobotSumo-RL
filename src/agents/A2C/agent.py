import torch
import torch.nn.functional as F

from src.agents.A2C.networks import ActorCriticNet


class A2CAgent:
    def __init__(self, obs_size, lr, device, gamma, entropy_coef):
        self.model = ActorCriticNet(obs_size=obs_size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef

    def compute_loss(self, ep_data):
        returns = []
        R = 0
        for r in reversed(ep_data["rews"]):
            R = r + self.gamma * R
            returns.insert(0, R)

        ret = torch.tensor(returns, device=self.device, dtype=torch.float32).view(-1, 1)
        lps = torch.cat(ep_data["lps"]).view(-1, 1)
        vals = torch.cat(ep_data["vals"]).view(-1, 1)
        ents = torch.cat(ep_data["ents"]).view(-1, 1)

        if ret.std() > 1e-5:
            ret = (ret - ret.mean()) / (ret.std() + 1e-8)

        advantage = ret - vals.detach()

        actor_loss = -(lps * advantage).mean()
        critic_loss = 0.5 * F.mse_loss(vals, ret)
        entropy_loss = -self.entropy_coef * ents.mean()

        return actor_loss + critic_loss + entropy_loss

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
