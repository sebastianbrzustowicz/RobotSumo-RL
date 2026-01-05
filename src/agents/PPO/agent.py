import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader, TensorDataset
from torch import distributions

class BackboneNetwork(nn.Module):
    def __init__(self, in_features, hidden_dimensions, out_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_dimensions)
        self.layer2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.layer3 = nn.Linear(hidden_dimensions, out_features)
        
    def forward(self, x):
        x = f.relu(self.layer1(x))
        x = f.relu(self.layer2(x))
        return self.layer3(x)

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic
        
    def forward(self, state):
        # Actor returns distribution parameters (mu and sigma)
        action_params = self.actor(state)
        value_pred = self.critic(state)
        return action_params, value_pred

def create_agent(input_features, hidden_dimensions):
    # Actor output: 2 for mu (mean) + 2 for sigma (deviation)
    actor = BackboneNetwork(input_features, hidden_dimensions, 4)
    critic = BackboneNetwork(input_features, hidden_dimensions, 1)
    return ActorCritic(actor, critic)

def get_distribution(action_params):
    mu, raw_sigma = torch.chunk(action_params, 2, dim=-1)
    mu = torch.tanh(mu)
    sigma = f.softplus(raw_sigma) + 1e-3
    return distributions.Normal(mu, sigma)

def calculate_returns(rewards, discount_factor, last_value, masks):
    returns = torch.zeros_like(torch.tensor(rewards))
    R = last_value
    for t in reversed(range(len(rewards))):
        R = rewards[t] + discount_factor * R * masks[t]
        returns[t] = R
    return returns

def calculate_surrogate_loss(old_log_probs, new_log_probs, epsilon, advantages):
    advantages = advantages.detach()
    policy_ratio = (new_log_probs - old_log_probs).exp()
    
    surr1 = policy_ratio * advantages
    surr2 = torch.clamp(policy_ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    return torch.min(surr1, surr2)

def calculate_losses(surr_loss, entropy, entropy_coeff, returns, value_pred):
    entropy_bonus = entropy_coeff * entropy
    policy_loss = -(surr_loss + entropy_bonus).mean()
    value_loss = f.mse_loss(value_pred, returns)
    return policy_loss, value_loss

def collect_trajectories(env, agent, discount_factor, device):
    states, actions, log_probs, values, rewards, masks = [], [], [], [], [], []
    state = env.reset()
    if isinstance(state, tuple): state = state[0]
    
    done = False
    episode_reward = 0
    
    while not done:
        state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
        states.append(state_tensor)

        action_params, value_pred = agent(state_tensor)
        dist = get_distribution(action_params)
        
        action = dist.sample()
        action_clipped = torch.clamp(action, -1.0, 1.0)
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        next_obs, reward, done, info = env.step(action_clipped.cpu().numpy()[0])
        if isinstance(next_obs, tuple): next_obs = next_obs[0]
        
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value_pred)
        rewards.append(reward)
        masks.append(1.0 - float(done))
        
        state = next_obs
        episode_reward += reward

    # Prepare data for training
    states = torch.cat(states)
    actions = torch.cat(actions)
    log_probs = torch.cat(log_probs).detach()
    values = torch.cat(values).squeeze(-1).detach()
    
    with torch.no_grad():
        _, last_value = agent(torch.FloatTensor(state).to(device).unsqueeze(0))
        last_value = last_value.item()

    returns = calculate_returns(rewards, discount_factor, last_value, masks).to(device)
    advantages = (returns - values)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return episode_reward, states, actions, log_probs, advantages, returns

def update_policy(agent, optimizer, states, actions, old_log_probs, advantages, returns, ppo_steps, epsilon, entropy_coeff):
    dataset = DataLoader(
        TensorDataset(states, actions, old_log_probs, advantages, returns),
        batch_size=64, shuffle=True
    )

    for _ in range(ppo_steps):
        for batch in dataset:
            b_states, b_actions, b_old_probs, b_adv, b_ret = batch
            
            action_params, value_pred = agent(b_states)
            value_pred = value_pred.squeeze(-1)
            
            dist = get_distribution(action_params)
            new_log_probs = dist.log_prob(b_actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            surr_loss = calculate_surrogate_loss(b_old_probs, new_log_probs, epsilon, b_adv)
            policy_loss, value_loss = calculate_losses(surr_loss, entropy, entropy_coeff, b_ret, value_pred)

            loss = policy_loss + 0.5 * value_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()