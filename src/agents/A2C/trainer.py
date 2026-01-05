import glob
import os
import random
from collections import deque

import pygame
import torch

from src.agents.A2C.agent import A2CAgent
from src.agents.A2C.networks import ActorCriticNet, select_action
from src.agents.A2C.rewards import get_reward
from src.env.sumo_env import SumoEnv

# --- FULL CONFIGURATION ---
cfg = {
    "lr": 3e-4,  # Reduced from 1e-3 for A2C stability
    "gamma": 0.99,  # Discount factor
    "entropy_coef": 0.01,  # Entropy weight (encourages exploration)
    "episodes": 100000,
    "max_steps": 1000,
    "render": False,
    "model_dir": "models",
    "master_path": "models/sumo_push_master.pt",
}

HISTORY_DIR = os.path.join(cfg["model_dir"], "history/A2C")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_history_models():
    return glob.glob(os.path.join(HISTORY_DIR, "model_v*.pt"))


def train():
    os.makedirs(HISTORY_DIR, exist_ok=True)

    env = SumoEnv(render_mode="human" if cfg["render"] else None)

    agent = A2CAgent(
        obs_size=11,
        lr=cfg["lr"],
        device=DEVICE,
        gamma=cfg["gamma"],
        entropy_coef=cfg["entropy_coef"],
    )

    opp_net = ActorCriticNet(obs_size=11).to(DEVICE).eval()

    if os.path.exists(cfg["master_path"]):
        agent.load(cfg["master_path"])
        print(f"📁 Loaded MASTER")
    else:
        agent.save(os.path.join(HISTORY_DIR, "model_v0.pt"))
        agent.save(cfg["master_path"])

    win_history = deque(maxlen=100)
    last_update_ep = 0
    print(
        f"🚀 Start A2C Training | LR: {cfg['lr']} | Gamma: {cfg['gamma']} | Device: {DEVICE}"
    )

    for ep in range(cfg["episodes"]):
        history_files = get_history_models()
        is_fighting_master = random.random() >= 0.20 or not history_files
        path = (
            cfg["master_path"] if is_fighting_master else random.choice(history_files)
        )
        opp_net.load_state_dict(torch.load(path, map_location=DEVICE))
        opp_name = "MASTER" if is_fighting_master else os.path.basename(path)

        obs = env.reset(randPositions=True)
        ep_data = {"lps": [], "vals": [], "rews": [], "ents": [], "reward_sum": 0.0}

        for step in range(cfg["max_steps"]):
            if cfg["render"]:
                env.render()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return

            act_ai, lp, ent, val = select_action(agent.model, obs[0], DEVICE)
            with torch.no_grad():
                act_opp, _, _, _ = select_action(opp_net, obs[1], DEVICE)

            next_obs, _, env_done, info = env.step(act_ai.flatten(), act_opp.flatten())

            done = env_done or (step == cfg["max_steps"] - 1)
            reward = get_reward(
                None, info, done, next_obs[0], info.get("is_collision", False)
            )

            ep_data["lps"].append(lp)
            ep_data["vals"].append(val)
            ep_data["ents"].append(ent)
            ep_data["rews"].append(reward)
            ep_data["reward_sum"] += reward

            obs = next_obs

            if done:
                winner = info.get("winner", 0)
                if is_fighting_master:
                    win_history.append(
                        1.0 if winner == 1 else (0.5 if winner == 0 else 0.0)
                    )

                wr = sum(win_history) / len(win_history) if win_history else 0
                status = "WIN" if winner == 1 else ("LOSE" if winner == 2 else "DRAW")
                print(
                    f"Ep {ep+1:04d} | Steps: {step+1:4} | vs {opp_name:12} | Reward: {ep_data['reward_sum']:7.2f} | WR: {wr:.2%} | {status}"
                )
                break

        if len(ep_data["rews"]) > 1:
            loss = agent.compute_loss(ep_data)
            agent.update(loss)

        # Master Update Logic
        wr = sum(win_history) / len(win_history) if win_history else 0
        draw_count = sum(1 for score in win_history if score == 0.5)

        c_list = [(0.51, 40, 50), (0.55, 24, 46), (0.60, 5, 40)]

        update_triggered = False
        if len(win_history) >= 100:
            for threshold_wr, wait_ep, max_draws in c_list:
                if (
                    wr >= threshold_wr
                    and (ep - last_update_ep) >= wait_ep
                    and draw_count < max_draws
                ):
                    update_triggered = True
                    break

        if update_triggered:
            ver = len(get_history_models())
            agent.save(cfg["master_path"])
            agent.save(os.path.join(HISTORY_DIR, f"model_v{ver}.pt"))
            last_update_ep = ep
            print(f"🔥 [NEW MASTER] v{ver} WR: {wr:.2%} | Ep: {ep}")

    pygame.quit()


if __name__ == "__main__":
    train()
