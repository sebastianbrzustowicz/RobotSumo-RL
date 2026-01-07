import os
import sys

import matplotlib.pyplot as plt
import pygame
import torch

from src.agents.A2C.networks import ActorCriticNet
from src.agents.A2C.networks import select_action as a2c_select
from src.agents.A2C.rewards import get_reward
from src.agents.PPO.agent import create_agent as create_ppo_agent
from src.agents.SAC.networks import GaussianActor as SACActor
from src.env.sumo_env import SumoEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PLAYER CONFIGURATION ---
# Types: "ai" / "human" / "dummy"
# Arch:  "a2c" / "ppo" / "sac"

PLAYER_1_TYPE = "ai"
PLAYER_1_ARCH = "sac"
MODEL_1_PATH = "models/favourite/SAC/model_v23.pt"

PLAYER_2_TYPE = "ai"
PLAYER_2_ARCH = "sac"
MODEL_2_PATH = "models/history/SAC/model_v25.pt"

MAX_STEPS = 1000

# --- PLOT INITIALIZATION ---
plt.ion()
fig, ax = plt.subplots(figsize=(8, 4))
(line1,) = ax.plot([], [], "g-", label="Robot 1 (Green)", linewidth=1.5)
(line2,) = ax.plot([], [], "b-", label="Robot 2 (Blue)", linewidth=1.5)
ax.set_title("Cumulative Reward")
ax.set_xlabel("Step")
ax.set_ylabel("Total Reward")
ax.legend()
ax.grid(True, alpha=0.3)


def load_ai_model(path, arch, device):
    if not os.path.exists(path):
        print(f"Model not found: {path}")
        return None

    try:
        sd = torch.load(path, map_location=device)

        if arch == "a2c":
            model = ActorCriticNet(obs_size=11).to(device)
            model.load_state_dict(sd)
        elif arch == "ppo":
            model = create_ppo_agent(11, 128).to(device)
            model.load_state_dict(sd)
        elif arch == "sac":
            model = SACActor(obs_size=11, action_dim=2).to(device)

            actor_sd = {}
            if any(k.startswith("actor.") for k in sd.keys()):
                for k, v in sd.items():
                    if k.startswith("actor."):
                        actor_sd[k[6:]] = v
            else:
                actor_sd = sd
            model.load_state_dict(actor_sd)
        else:
            return None

        model.eval()
        return model
    except Exception as e:
        print(f"Loading error {arch.upper()}: {e}")
        return None


def get_action(p_type, arch, robot_idx, state, model):
    if p_type == "dummy":
        return [0.0, 0.0]
    if p_type == "human":
        keys = pygame.key.get_pressed()
        v, omega = 0.0, 0.0
        if robot_idx == 0:
            if keys[pygame.K_UP]:
                v = 1.0
            if keys[pygame.K_DOWN]:
                v = -1.0
            if keys[pygame.K_LEFT]:
                omega = 1.0
            if keys[pygame.K_RIGHT]:
                omega = -1.0
        else:
            if keys[pygame.K_w]:
                v = 1.0
            if keys[pygame.K_s]:
                v = -1.0
            if keys[pygame.K_a]:
                omega = 1.0
            if keys[pygame.K_d]:
                omega = -1.0
        return [v, omega]

    if p_type == "ai" and model:
        obs_vec = state[robot_idx]
        with torch.no_grad():
            if arch == "a2c":
                act, _, _, _ = a2c_select(model, obs_vec, DEVICE)
                return act.flatten()
            elif arch == "ppo":
                obs_t = torch.FloatTensor(obs_vec).to(DEVICE).unsqueeze(0)
                action_params, _ = model(obs_t)
                mu, _ = torch.chunk(action_params, 2, dim=-1)
                return torch.tanh(mu).cpu().numpy().flatten()
            elif arch == "sac":
                obs_t = torch.FloatTensor(obs_vec).to(DEVICE).unsqueeze(0)
                _, _, mu = model.sample(obs_t)
                return mu.cpu().numpy().flatten()
    return [0.0, 0.0]


def main():
    env = SumoEnv(render_mode=True)
    m1 = (
        load_ai_model(MODEL_1_PATH, PLAYER_1_ARCH, DEVICE)
        if PLAYER_1_TYPE == "ai"
        else None
    )
    m2 = (
        load_ai_model(MODEL_2_PATH, PLAYER_2_ARCH, DEVICE)
        if PLAYER_2_TYPE == "ai"
        else None
    )

    scores = [0, 0]
    round_count = 0

    print(f"\nCROSS-PLAY: {PLAYER_1_ARCH.upper()} vs {PLAYER_2_ARCH.upper()}")

    while True:
        state = env.reset(randPositions=True)
        if isinstance(state, tuple):
            state = state[0]

        done, step_count = False, 0
        total_r1, total_r2 = 0.0, 0.0
        steps_h, r1_h, r2_h = [], [], []

        while not done and step_count < MAX_STEPS:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    plt.close()
                    pygame.quit()
                    return

            act1 = get_action(PLAYER_1_TYPE, PLAYER_1_ARCH, 0, state, m1)
            act2 = get_action(PLAYER_2_TYPE, PLAYER_2_ARCH, 1, state, m2)

            state, _, env_done, info = env.step(act1, act2)
            if isinstance(state, tuple):
                state = state[0]

            done = env_done or (step_count + 1 >= MAX_STEPS)

            # Perspective for robot 2
            info_r2 = info.copy()
            if info.get("winner") == 1:
                info_r2["winner"] = 2
            elif info.get("winner") == 2:
                info_r2["winner"] = 1

            r1_s = get_reward(
                None, info, done, state[0], info.get("is_collision", False)
            )
            r2_s = get_reward(
                None, info_r2, done, state[1], info.get("is_collision", False)
            )

            total_r1 += r1_s
            total_r2 += r2_s
            steps_h.append(step_count)
            r1_h.append(total_r1)
            r2_h.append(total_r2)

            if step_count % 10 == 0:
                line1.set_data(steps_h, r1_h)
                line2.set_data(steps_h, r2_h)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.flush_events()

            sys.stdout.write(
                f"\rStep: {step_count:4d} | R1 ({PLAYER_1_ARCH}): {total_r1:7.1f} | R2 ({PLAYER_2_ARCH}): {total_r2:7.1f}"
            )
            sys.stdout.flush()

            step_count += 1
            env.render(
                names=[os.path.basename(MODEL_1_PATH), os.path.basename(MODEL_2_PATH)],
                archs=[PLAYER_1_ARCH.upper(), PLAYER_2_ARCH.upper()],
            )

        round_count += 1
        winner = info.get("winner", 0)
        if winner == 1:
            scores[0] += 1
        elif winner == 2:
            scores[1] += 1
        print(
            f"\n--- Round {round_count} Over. Winner: {winner} | Score: {scores[0]}-{scores[1]} ---"
        )

        pygame.time.wait(1000)
        if PLAYER_1_TYPE == "ai":
            m1 = load_ai_model(MODEL_1_PATH, PLAYER_1_ARCH, DEVICE)
        if PLAYER_2_TYPE == "ai":
            m2 = load_ai_model(MODEL_2_PATH, PLAYER_2_ARCH, DEVICE)


if __name__ == "__main__":
    main()
