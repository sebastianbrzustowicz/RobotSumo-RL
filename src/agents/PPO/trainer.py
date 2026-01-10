import glob
import os
import random
from collections import deque

import pygame
import torch
import torch.optim as optim

from src.agents.PPO.agent import (
    calculate_returns,
    create_agent,
    get_distribution,
    update_policy,
)
from src.agents.PPO.rewards import get_reward
from src.env.sumo_env import SumoEnv

cfg = {
    "lr": 3e-4,
    "gamma": 0.99,
    "ppo_epochs": 10,  # Number of optimization epochs per update cycle
    "eps_clip": 0.2,  # Clipping parameter for the policy objective to prevent large updates
    "entropy_coef": 0.01,  # Weight of the entropy bonus to encourage exploration
    "update_every_steps": 2048,  # Total steps collected before performing an update
    "max_steps": 1000,
    "episodes": 100000,
    "render": False,
    "master_path": "models/ppo_push_master.pt",
    "model_dir": "models/",
}


def get_history_models(dir):
    return glob.glob(os.path.join(dir, "model_v*.pt"))


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history_dir = os.path.join(cfg["model_dir"], "history/PPO")
    os.makedirs(history_dir, exist_ok=True)

    model = create_agent(11, 128).to(device)

    if os.path.exists(cfg["master_path"]):
        model.load_state_dict(torch.load(cfg["master_path"], map_location=device))
        print("📁 Loaded MASTER model")
    else:
        print("🆕 Initializing new master.")
        torch.save(model.state_dict(), cfg["master_path"])

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    opp_net = create_agent(11, 128).to(device).eval()

    win_history = deque(maxlen=100)
    last_update_ep = 0
    buffer_steps = 0
    storage = {"s": [], "a": [], "lp": [], "ad": [], "rt": []}
    env = SumoEnv(render_mode=cfg["render"])

    # Master update thresholds (WR, min_ep_break, max_draws)
    c_list = [
        (0.51, 40, 50),
        (0.52, 36, 49),
        (0.53, 32, 48),
        (0.54, 28, 47),
        (0.55, 24, 46),
        (0.56, 20, 45),
        (0.57, 16, 44),
        (0.58, 12, 43),
        (0.59, 8, 42),
        (0.60, 5, 40),
    ]

    try:
        for ep in range(cfg["episodes"]):
            hist = get_history_models(history_dir)
            is_master = random.random() >= 0.20 or not hist
            opp_path = cfg["master_path"] if is_master else random.choice(hist)
            opp_name = "MASTER" if is_master else os.path.basename(opp_path)

            opp_net.load_state_dict(torch.load(opp_path, map_location=device))

            state_vecs = env.reset(randPositions=True)
            done, ep_rew, ep_steps = False, 0, 0
            ep_data = {"s": [], "a": [], "lp": [], "v": [], "r": [], "m": []}

            while not done:
                if cfg["render"]:
                    env.render()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return

                s_t = torch.as_tensor(
                    state_vecs[0], dtype=torch.float32, device=device
                ).unsqueeze(0)
                opp_s_t = torch.as_tensor(
                    state_vecs[1], dtype=torch.float32, device=device
                ).unsqueeze(0)

                with torch.no_grad():
                    a_p, v_p = model(s_t)
                    dist = get_distribution(a_p)
                    act = dist.sample()
                    o_p, _ = opp_net(opp_s_t)
                    o_act = get_distribution(o_p).sample()

                act_np = torch.clamp(act, -1, 1).cpu().numpy()[0]
                o_act_np = torch.clamp(o_act, -1, 1).cpu().numpy()[0]

                next_state_vecs, _, env_done, info = env.step(act_np, o_act_np)

                ep_steps += 1
                if ep_steps >= cfg.get("max_steps", 1000):
                    done = True
                    info["winner"] = 0  # Time is out
                else:
                    done = env_done

                rew = get_reward(
                    None,
                    info,
                    done,
                    next_state_vecs[0],
                    info.get("is_collision", False),
                )

                # Save the data
                ep_data["s"].append(s_t)
                ep_data["a"].append(act)
                ep_data["lp"].append(dist.log_prob(act).sum(-1))
                ep_data["v"].append(v_p)
                ep_data["r"].append(rew)
                ep_data["m"].append(1.0 - float(done))

                state_vecs, ep_rew, buffer_steps = (
                    next_state_vecs,
                    ep_rew + rew,
                    buffer_steps + 1,
                )

            # Calculate returns for PPO
            with torch.no_grad():
                _, last_v = model(
                    torch.as_tensor(
                        state_vecs[0], dtype=torch.float32, device=device
                    ).unsqueeze(0)
                )

            rets = calculate_returns(
                ep_data["r"], cfg["gamma"], last_v.item(), ep_data["m"]
            ).to(device)
            advs = rets - torch.cat(ep_data["v"]).squeeze(-1)

            storage["s"].append(torch.cat(ep_data["s"]))
            storage["a"].append(torch.cat(ep_data["a"]))
            storage["lp"].append(torch.cat(ep_data["lp"]))
            storage["ad"].append(advs)
            storage["rt"].append(rets)

            # --- LOGIN AND STATISTICS ---
            winner = info.get("winner", 0)
            if is_master:
                win_history.append(
                    1.0 if winner == 1 else (0.5 if winner == 0 else 0.0)
                )

            wr = sum(win_history) / len(win_history) if win_history else 0
            status = "WIN" if winner == 1 else ("LOSE" if winner == 2 else "DRAW")
            print(
                f"Ep {ep:04d} | Steps: {ep_steps:4} | vs {opp_name:12} | Reward: {ep_rew:7.2f} | WR: {wr:.2%} | {status}"
            )

            # --- NETWORK UPDATE (PPO) ---
            if buffer_steps >= cfg["update_every_steps"]:
                adv_b = torch.cat(storage["ad"]).detach()
                update_policy(
                    model,
                    optimizer,
                    torch.cat(storage["s"]),
                    torch.cat(storage["a"]),
                    torch.cat(storage["lp"]).detach(),
                    (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8),
                    torch.cat(storage["rt"]).detach(),
                    cfg["ppo_epochs"],
                    cfg["eps_clip"],
                    cfg["entropy_coef"],
                )
                storage = {k: [] for k in storage}
                buffer_steps = 0
                print("--- [Policy Updated] ---")

            # --- MASTER UPDATE LOGIC ---
            if len(win_history) >= 100:
                draw_count = sum(1 for score in win_history if score == 0.5)
                update_triggered = False
                for threshold_wr, wait_ep, max_draws in c_list:
                    if (
                        wr >= threshold_wr
                        and (ep - last_update_ep) >= wait_ep
                        and draw_count < max_draws
                    ):
                        update_triggered = True
                        break

                if update_triggered:
                    ver = len(get_history_models(history_dir))
                    torch.save(model.state_dict(), cfg["master_path"])
                    torch.save(
                        model.state_dict(),
                        os.path.join(history_dir, f"model_v{ver}.pt"),
                    )
                    last_update_ep = ep
                    print(
                        f"🔥 [NEW MASTER] v{ver} WR: {wr:.2%} | Draws: {draw_count} | Ep: {ep}"
                    )

    finally:
        pygame.quit()


if __name__ == "__main__":
    train()
