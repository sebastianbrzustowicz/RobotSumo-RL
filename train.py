import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import random
import glob
from collections import deque
from env.sumo_env import SumoEnv
from model import ActorCriticNet, select_action
from rewards import get_reward

EPISODES = 100000
MAX_STEPS = 1000
GAMMA = 0.99
LR = 1e-3
RENDER = False 
MODEL_DIR = "models"
HISTORY_DIR = os.path.join(MODEL_DIR, "history")
MASTER_PATH = os.path.join(MODEL_DIR, "sumo_push_master.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_history_models():
    return glob.glob(os.path.join(HISTORY_DIR, "model_v*.pt"))

def train():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    env = SumoEnv(render_mode=RENDER)
    
    model = ActorCriticNet(obs_size=11).to(DEVICE)
    if os.path.exists(MASTER_PATH):
        model.load_state_dict(torch.load(MASTER_PATH))
        print(f"📁 Loaded: {MASTER_PATH}")
    else:
        torch.save(model.state_dict(), os.path.join(HISTORY_DIR, "model_v0.pt"))
        torch.save(model.state_dict(), MASTER_PATH)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    opponent_model = ActorCriticNet(obs_size=11).to(DEVICE)
    opponent_model.eval()
    
    win_history = deque(maxlen=100)
    last_update_ep = 0
    print(f"🚀 Start: {DEVICE}")

    for ep in range(EPISODES):
        history_files = get_history_models()
        is_fighting_master = True
        
        if random.random() < 0.20 and len(history_files) > 0:
            chosen_opp_path = random.choice(history_files)
            is_fighting_master = False
        else:
            chosen_opp_path = MASTER_PATH
            
        opponent_model.load_state_dict(torch.load(chosen_opp_path))
        opp_name = "MASTER" if is_fighting_master else os.path.basename(chosen_opp_path)

        all_obs = env.reset(randPositions=True)
        state_ai = all_obs[0]   
        done = False
        had_collision = False 
        episode_reward = 0.0
        log_probs, values, rewards, entropies = [], [], [], []
        
        for step in range(MAX_STEPS):
            action_np, lp, ent, val = select_action(model, state_ai, DEVICE)
            action_env = action_np.flatten()
            
            with torch.no_grad():
                act_opp_np, _, _, _ = select_action(opponent_model, all_obs[1], DEVICE)
                action_enemy = act_opp_np.flatten()
            
            next_all_obs, _, done, info = env.step(action_env, action_enemy)
            if info.get('collision', False): had_collision = True
            
            next_state_ai = next_all_obs[0]
            reward = get_reward(env, info, done, next_state_ai, had_collision)
            
            log_probs.append(lp)
            values.append(val)
            rewards.append(reward)
            entropies.append(ent)
            
            episode_reward += reward
            state_ai = next_state_ai
            all_obs = next_all_obs 
            if RENDER: env.render()
            if done: break

        if len(rewards) > 1:
            R = 0
            returns = []
            for r in reversed(rewards):
                R = r + GAMMA * R
                returns.insert(0, R)
            
            returns = torch.tensor(returns, device=DEVICE, dtype=torch.float32).view(-1, 1)
            log_probs = torch.cat(log_probs).view(-1, 1)
            values = torch.cat(values).view(-1, 1)
            
            if returns.std() > 1e-5:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            advantage = returns - values.detach()
            actor_loss = -(log_probs * advantage).mean()
            critic_loss = F.mse_loss(values, returns)
            entropy_loss = -0.01 * torch.cat(entropies).mean()
            
            loss = actor_loss + 0.5 * critic_loss + entropy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        winner = info.get('winner', 0)
        
        if is_fighting_master:
            if winner == 1:
                score = 1.0
            elif winner == 0:
                score = 0.5
            else:
                score = 0.0
            win_history.append(score)
        
        current_win_rate = sum(win_history) / len(win_history) if len(win_history) > 0 else 0
        status = "WIN" if winner == 1 else ("LOSE" if winner == 2 else "DRAW")
        
        print(f"Ep {ep+1:04d} | vs {opp_name:12} | Total Reward: {episode_reward:7.2f} | WR: {current_win_rate:.2%} | {status}")

        condition_1 = len(win_history) >= 20 and current_win_rate >= 0.52 and (ep - last_update_ep) >= 50
        condition_2 = len(win_history) >= 20 and current_win_rate >= 0.54 and (ep - last_update_ep) >= 30
        condition_3 = len(win_history) >= 20 and current_win_rate >= 0.56 and (ep - last_update_ep) >= 15

        if condition_1 or condition_2 or condition_3:
            version = len(get_history_models())
            new_v_path = os.path.join(HISTORY_DIR, f"model_v{version}.pt")
            torch.save(model.state_dict(), new_v_path)
            torch.save(model.state_dict(), MASTER_PATH)
            last_update_ep = ep
            print(f"🔥 [NEW MASTER] v{version} WR: {current_win_rate:.2%}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "final_model.pt"))
    print("✅ The end.")

if __name__ == "__main__":
    train()