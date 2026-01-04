import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import random
import glob
import pygame
from collections import deque
from env.sumo_env import SumoEnv
from src.agents.ppo.networks import ActorCriticNet, select_action
from src.agents.ppo.rewards import get_reward
from src.common.mlflow_utils import MLflowManager

def get_history_models(history_dir):
    return glob.glob(os.path.join(history_dir, "model_v*.pt"))

def train(config_path="configs/ppo_v1.yaml"):
    # --- INITIALIZE MLFLOW & CONFIG ---
    ml_manager = MLflowManager(experiment_name="Sumo_PPO_SelfPlay")
    cfg = ml_manager.load_config(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history_dir = os.path.join(cfg['model_dir'], "history")
    os.makedirs(history_dir, exist_ok=True)

    # Begin MLflow session
    ml_manager.start_run(run_name="PPO_SelfPlay_Base", config=cfg)

    # --- ENVIRONMENT & MODEL SETUP ---
    env = SumoEnv(render_mode=cfg['render'])
    model = ActorCriticNet(obs_size=11).to(device)
    
    if os.path.exists(cfg['master_path']):
        model.load_state_dict(torch.load(cfg['master_path']))
        print(f"📁 Loaded MASTER model")
    else:
        torch.save(model.state_dict(), cfg['master_path'])

    optimizer = optim.Adam(model.parameters(), lr=float(cfg['lr']))
    opponent_net = ActorCriticNet(obs_size=11).to(device).eval()
    
    win_history = deque(maxlen=100)
    last_update_ep = 0
    b_states, b_actions, b_lps, b_vals, b_rews, b_masks = [], [], [], [], [], []
    current_buffer_steps = 0

    print(f"🚀 Training started on {device}")

    try:
        for ep in range(cfg['episodes']):
            # Render events
            if cfg['render']:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: return

            # Opponent selection
            history_files = get_history_models(history_dir)
            is_fighting_master = random.random() >= 0.20 or len(history_files) == 0
            opp_path = cfg['master_path'] if is_fighting_master else random.choice(history_files)
            
            try:
                opponent_net.load_state_dict(torch.load(opp_path))
            except: pass
                
            obs = env.reset(randPositions=True)
            episode_reward = 0
            
            for step in range(cfg['max_steps']):
                if cfg['render']: env.render()
                
                state_tensor = torch.tensor(obs[0], dtype=torch.float32, device=device)
                
                with torch.no_grad():
                    act_ai, lp, _, val = select_action(model, obs[0], device)
                    act_opp, _, _, _ = select_action(opponent_net, obs[1], device)
                
                next_obs, _, done, info = env.step(act_ai.flatten(), act_opp.flatten())
                reward = get_reward(None, info, done, next_obs[0], info.get('is_collision', False))
                
                b_states.append(state_tensor)
                b_actions.append(torch.tensor(act_ai, device=device).flatten())
                b_lps.append(lp.detach())
                b_vals.append(val.detach())
                b_rews.append(torch.tensor([reward], device=device))
                b_masks.append(torch.tensor([1.0 - float(done)], device=device))
                
                obs = next_obs
                episode_reward += reward
                current_buffer_steps += 1

                if done:
                    winner = info.get('winner', 0)
                    if is_fighting_master:
                        win_history.append(1.0 if winner == 1 else (0.5 if winner == 0 else 0.0))
                    
                    wr = sum(win_history)/len(win_history) if win_history else 0
                    
                    # Logowanie metryk epizodu
                    ml_manager.log_metrics(ep, {
                        "reward": episode_reward,
                        "win_rate": wr,
                        "steps": step + 1
                    })
                    
                    if ep % 10 == 0:
                        print(f"Ep {ep:04d} | Rew: {episode_reward:6.2f} | WR: {wr:.1%}")
                    break

            # --- PPO UPDATE LOGIC ---
            if current_buffer_steps >= cfg['update_every_steps']:
                with torch.no_grad():
                    _, _, next_val = model(torch.tensor(obs[0], dtype=torch.float32, device=device))
                
                v = torch.cat(b_vals).view(-1)
                r = torch.cat(b_rews).view(-1)
                m = torch.cat(b_masks).view(-1)
                
                returns = torch.zeros_like(r)
                advantages = torch.zeros_like(r)
                last_gae_lam = 0
                
                for t in reversed(range(len(r))):
                    next_v = next_val if t == len(r) - 1 else v[t+1]
                    delta = r[t] + cfg['gamma'] * next_v * m[t] - v[t]
                    advantages[t] = last_gae_lam = delta + cfg['gamma'] * cfg['gae_lambda'] * m[t] * last_gae_lam
                
                returns = advantages + v
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                S, A = torch.stack(b_states), torch.stack(b_actions)
                OLD_LP, RET, ADV = torch.cat(b_lps).view(-1, 1), returns.view(-1, 1), advantages.view(-1, 1)

                for _ in range(cfg['ppo_epochs']):
                    mu, std, v_curr = model(S)
                    dist = torch.distributions.Normal(mu, std)
                    new_lp = dist.log_prob(A).sum(dim=-1, keepdim=True)
                    entropy = dist.entropy().sum(dim=-1, keepdim=True)
                    
                    ratio = torch.exp(new_lp - OLD_LP)
                    surr1 = ratio * ADV
                    surr2 = torch.clamp(ratio, 1.0 - cfg['eps_clip'], 1.0 + cfg['eps_clip']) * ADV
                    
                    loss_pi = -torch.min(surr1, surr2).mean()
                    loss_v = 0.5 * F.mse_loss(v_curr, RET)
                    loss_ent = -cfg['entropy_coef'] * entropy.mean()
                    
                    loss = loss_pi + loss_v + loss_ent
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                # Logowanie metryk PPO
                ml_manager.log_metrics(ep, {
                    "loss_total": loss.item(),
                    "loss_policy": loss_pi.item(),
                    "loss_value": loss_v.item(),
                    "entropy": entropy.mean().item()
                })

                b_states, b_actions, b_lps, b_vals, b_rews, b_masks = [], [], [], [], [], []
                current_buffer_steps = 0
                print("✨ Model Updated & Logged")

            # Promotion logic
            wr = sum(win_history)/len(win_history) if win_history else 0
            if len(win_history) >= 100 and (ep - last_update_ep) >= 100 and wr >= 0.55:
                ver = len(get_history_models(history_dir))
                torch.save(model.state_dict(), cfg['master_path'])
                torch.save(model.state_dict(), os.path.join(history_dir, f"model_v{ver}.pt"))
                last_update_ep = ep
                print(f"🔥 NEW MASTER v{ver}")

    finally:
        ml_manager.end_run()
        env.close()

if __name__ == "__main__":
    train()