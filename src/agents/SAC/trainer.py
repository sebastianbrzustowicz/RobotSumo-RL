import glob
import os
import random
import sys
from collections import deque

import pygame
import torch
import numpy as np

from src.agents.SAC.agent import SACAgent, ReplayBuffer
from src.agents.SAC.rewards import get_reward
from src.agents.SAC.networks import GaussianActor
from src.env.sumo_env import SumoEnv

cfg = {
    "lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,           # Soft update coefficient for target networks
    "batch_size": 256,      # SAC performs better with larger batch sizes
    "buffer_capacity": 1000000,
    "start_steps": 2000,    # Initial random actions to seed the replay buffer
    "update_after": 1000,   # Minimum steps before starting the learning process
    "max_steps": 1000,
    "episodes": 100000,
    "render": False,
    "master_path": "models/sac_sumo_master.pt",
    "model_dir": "models/",
}

def get_history_models(dir):
    return glob.glob(os.path.join(dir, "model_v*.pt"))

def train():
    """Main training loop for the SAC agent."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history_dir = os.path.join(cfg["model_dir"], "history/SAC")
    os.makedirs(history_dir, exist_ok=True)

    agent = SACAgent(obs_size=11, action_dim=2, device=device, lr=cfg["lr"])
    memory = ReplayBuffer(cfg["buffer_capacity"])

    if os.path.exists(cfg["master_path"]):
        agent.load_state_dict(torch.load(cfg["master_path"], map_location=device))
        print("Loaded MASTER SAC model")
    else:
        print("Initializing new master.")
        torch.save(agent.state_dict(), cfg["master_path"])

    opp_net = GaussianActor(11, 2).to(device).eval()

    win_history = deque(maxlen=100)
    total_steps = 0
    last_update_ep = 0
    env = SumoEnv(render_mode=cfg["render"])

    # Thresholds: (WinRate, MinEpisodesBreak, MaxDraws)
    c_list = [(0.51, 12, 50), (0.55, 8, 46), (0.60, 4, 40)]

    try:
        for ep in range(cfg["episodes"]):
            hist = get_history_models(history_dir)
            is_master = random.random() >= 0.20 or not hist
            opp_path = cfg["master_path"] if is_master else random.choice(hist)
            sd = torch.load(opp_path, map_location=device)
            
            actor_sd = {}
            for k, v in sd.items():
                if k.startswith("actor."):
                    new_key = k[6:] 
                    actor_sd[new_key] = v
            
            if actor_sd:
                opp_net.load_state_dict(actor_sd)
            else:
                opp_net.load_state_dict(sd)

            obs = env.reset(randPositions=True)
            done, ep_rew, ep_steps = False, 0, 0

            while not done:
                if cfg["render"]:
                    env.render()
                
                if total_steps < cfg["start_steps"]:
                    act_np = np.random.uniform(-1, 1, 2)
                else:
                    s_t = torch.FloatTensor(obs[0]).to(device).unsqueeze(0)
                    with torch.no_grad():
                        action, _, _ = agent.actor.sample(s_t)
                    act_np = action.cpu().numpy()[0]

                opp_s_t = torch.FloatTensor(obs[1]).to(device).unsqueeze(0)
                with torch.no_grad():
                    _, _, opp_mu = opp_net.sample(opp_s_t)
                opp_act_np = opp_mu.cpu().numpy()[0]

                next_obs, _, env_done, info = env.step(act_np, opp_act_np)
                
                ep_steps += 1
                done = env_done or ep_steps >= cfg["max_steps"]
                if ep_steps >= cfg["max_steps"]: info["winner"] = 0

                rew = get_reward(None, info, done, next_obs[0], info.get("is_collision", False))
                
                memory.push(obs[0], act_np, rew, next_obs[0], float(done))

                # --- AGENT PARAMETERS UPDATE ---
                if len(memory) > cfg["batch_size"] and total_steps > cfg["update_after"]:
                    q_l, a_l, alpha_v = agent.update_parameters(memory, cfg["batch_size"], cfg["gamma"], cfg["tau"])

                obs = next_obs
                ep_rew += rew
                total_steps += 1

            winner = info.get("winner", 0)
            if is_master:
                win_history.append(1.0 if winner == 1 else (0.5 if winner == 0 else 0.0))
            
            wr = sum(win_history) / len(win_history) if win_history else 0
            sys.stdout.write(f"\rEp {ep:04d} | Steps: {ep_steps:4} | WR: {wr:.2%} | Rew: {ep_rew:7.2f} | Alpha: {alpha_v if 'alpha_v' in locals() else 0:.4f}")
            sys.stdout.flush()

            # --- MASTER UPDATE LOGIC ---
            if len(win_history) >= 20:
                draw_count = sum(1 for score in win_history if score == 0.5)
                if any(wr >= thr and (ep - last_update_ep) >= wait and draw_count < d for thr, wait, d in c_list):
                    ver = len(get_history_models(history_dir))
                    torch.save(agent.state_dict(), cfg["master_path"])
                    torch.save(agent.state_dict(), os.path.join(history_dir, f"model_v{ver}.pt"))
                    last_update_ep = ep
                    print(f"\n🔥 [NEW SAC MASTER] v{ver} WR: {wr:.2%}")

    finally:
        pygame.quit()

if __name__ == "__main__":
    train()