import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import os
import random
import glob
from collections import deque
from env.sumo_env import SumoEnv
from model import ActorCriticNet, select_action
from rewards import get_reward

# --- CONFIGURATION ---
NUM_ENVS = 12
EPISODES = 100000
MAX_STEPS = 1000
GAMMA = 0.99
LR = 1e-4         # PPO usually prefers slightly lower learning rates
RENDER = False 
MODEL_DIR = "models"
HISTORY_DIR = os.path.join(MODEL_DIR, "history")
MASTER_PATH = os.path.join(MODEL_DIR, "sumo_push_master.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PPO HYPERPARAMETERS ---
PPO_EPOCHS = 8    # Number of times to reuse the collected data
EPS_CLIP = 0.2    # Clipping range for the objective function
ENTROPY_COEF = 0.04

def make_env():
    return SumoEnv(render_mode=RENDER)

def env_worker(remote, env_fn):
    env = env_fn()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, rewards, done, info = env.step(data[0], data[1])
                if done:
                    obs = env.reset(randPositions=True)
                remote.send((obs, rewards, done, info))
            elif cmd == 'reset':
                remote.send(env.reset(randPositions=data))
            elif cmd == 'close':
                remote.close()
                break
        except EOFError:
            break

def get_history_models():
    return glob.glob(os.path.join(HISTORY_DIR, "model_v*.pt"))

def train():
    mp.set_start_method('spawn', force=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    
    remotes, work_remotes = zip(*[mp.Pipe() for _ in range(NUM_ENVS)])
    ps = [mp.Process(target=env_worker, args=(work_remotes[i], make_env)) 
          for i in range(NUM_ENVS)]
    for p in ps: 
        p.daemon = True
        p.start()

    model = ActorCriticNet(obs_size=11).to(DEVICE)
    if os.path.exists(MASTER_PATH):
        model.load_state_dict(torch.load(MASTER_PATH))
        print(f"📁 Loaded MASTER model: {MASTER_PATH}")
    else:
        torch.save(model.state_dict(), os.path.join(HISTORY_DIR, "model_v0.pt"))
        torch.save(model.state_dict(), MASTER_PATH)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    opponent_nets = [ActorCriticNet(obs_size=11).to(DEVICE).eval() for _ in range(NUM_ENVS)]
    
    win_history = deque(maxlen=100)
    last_update_ep = 0
    print(f"🚀 Start PPO Parallel Training: {DEVICE}")

    for ep in range(0, EPISODES, NUM_ENVS):
        history_files = get_history_models()
        batch_data = []

        for i in range(NUM_ENVS):
            is_fighting_master = random.random() >= 0.20 or len(history_files) == 0
            path = MASTER_PATH if is_fighting_master else random.choice(history_files)
            opponent_nets[i].load_state_dict(torch.load(path))
            
            remotes[i].send(('reset', True))
            all_obs = remotes[i].recv()
            
            batch_data.append({
                'is_master': is_fighting_master, 
                'opp_name': "MASTER" if is_fighting_master else os.path.basename(path),
                'obs': all_obs, 'reward': 0.0,
                'states': [], 'actions': [], 'lps': [], 'vals': [], 'rews': []
            })

        active_indices = list(range(NUM_ENVS))
        for step in range(MAX_STEPS):
            if not active_indices: break
            is_last_step = (step == MAX_STEPS - 1)
            current_step_indices = []
            
            for i in active_indices:
                d = batch_data[i]
                # In PPO we need to store the state and action to re-evaluate log_probs during epochs
                state_tensor = torch.tensor(d['obs'][0], dtype=torch.float32, device=DEVICE)
                act_ai, lp, _, val = select_action(model, d['obs'][0], DEVICE)
                
                with torch.no_grad():
                    act_opp, _, _, _ = select_action(opponent_nets[i], d['obs'][1], DEVICE)
                
                d['states'].append(state_tensor)
                d['actions'].append(torch.tensor(act_ai, device=DEVICE).flatten())
                d['lps'].append(lp.detach())
                d['vals'].append(val.detach())
                
                remotes[i].send(('step', (act_ai.flatten(), act_opp.flatten())))
                current_step_indices.append(i)

            for i in current_step_indices:
                next_obs, _, env_done, info = remotes[i].recv()
                done = env_done or is_last_step
                winner = info.get('winner', 0)
                
                is_coll = info.get('is_collision', False)
                reward = get_reward(None, info, done, next_obs[0], is_coll)
                reward = reward / 100.0

                batch_data[i]['rews'].append(reward)
                batch_data[i]['reward'] += reward
                batch_data[i]['obs'] = next_obs
                
                if done:
                    if batch_data[i]['is_master']:
                        win_history.append(1.0 if winner == 1 else (0.5 if winner == 0 else 0.0))
                    
                    wr = sum(win_history)/len(win_history) if win_history else 0
                    status = "WIN" if winner == 1 else ("LOSE" if winner == 2 else "DRAW")
                    print(f"Ep {ep+i+1:04d} | Steps: {step+1:4} | vs {batch_data[i]['opp_name']:12} | Reward: {batch_data[i]['reward']:7.2f} | WR: {wr:.2%} | {status}")
                    active_indices.remove(i)

        # --- PPO UPDATE LOGIC ---
        for i in range(NUM_ENVS):
            d = batch_data[i]
            if len(d['rews']) < 2: continue

            # Calculate Returns
            R = 0
            returns = []
            for r in reversed(d['rews']):
                R = r + GAMMA * R
                returns.insert(0, R)
            
            # Prepare tensors
            old_states = torch.stack(d['states'])
            old_actions = torch.stack(d['actions'])
            old_log_probs = torch.cat(d['lps']).detach()
            old_values = torch.cat(d['vals']).detach()
            returns = torch.tensor(returns, device=DEVICE, dtype=torch.float32).view(-1, 1)
            
            # Normalize advantages
            advantages = returns - old_values
            if advantages.std() > 1e-5:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO Epochs
            for _ in range(PPO_EPOCHS):
                # Re-evaluate same actions with current model state
                mu, std, current_values = model(old_states)
                dist = torch.distributions.Normal(mu, std)
                
                new_log_probs = dist.log_prob(old_actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1, keepdim=True)
                
                # Ratio for clipping
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Clipped Objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * F.mse_loss(current_values, returns)
                entropy_loss = -ENTROPY_COEF * entropy.mean()
                
                loss = policy_loss + value_loss + entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

        # --- MASTER MODEL PROMOTION LOGIC ---
        wr = sum(win_history)/len(win_history) if win_history else 0
        draw_count = sum(1 for score in win_history if score == 0.5)
        
        c_list = [
            (0.51, 40, 50), # WR >= 51%, min 40 ep break, max 10 draws in history
            (0.52, 36, 49),
            (0.53, 32, 48),
            (0.54, 28, 47),
            (0.55, 24, 46),
            (0.56, 20, 45),
            (0.57, 16, 44),
            (0.58, 12, 43),
            (0.59, 8, 42),
            (0.60, 5, 40)
        ]
        
        update_triggered = False
        if len(win_history) >= 100:
            for threshold_wr, wait_ep, max_draws in c_list:
                if wr >= threshold_wr and (ep - last_update_ep) >= wait_ep and draw_count < max_draws:
                    update_triggered = True
                    break

        if update_triggered:
            ver = len(get_history_models())
            torch.save(model.state_dict(), MASTER_PATH)
            torch.save(model.state_dict(), os.path.join(HISTORY_DIR, f"model_v{ver}.pt"))
            last_update_ep = ep
            print(f"🔥 [NEW PPO MASTER] v{ver} WR: {wr:.2%} | Ep: {ep}")

    for r in remotes: r.send(('close', None))
    for p in ps: p.join()

if __name__ == "__main__":
    train()