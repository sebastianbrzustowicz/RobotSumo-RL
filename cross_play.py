import os
import sys

import matplotlib.pyplot as plt
import pygame
import torch

from src.agents.A2C.networks import ActorCriticNet, select_action
from src.agents.A2C.rewards import get_reward
from src.agents.PPO.agent import create_agent
from src.env.sumo_env import SumoEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PLAYER CONFIGURATION ---
# Types: "ai" / "human" / "dummy"
# Architecture:  "a2c" / "ppo"

PLAYER_1_TYPE = "human"
PLAYER_1_ARCH = "a2c"
MODEL_1_PATH = "models/favourite/A2C/model_v70.pt"

PLAYER_2_TYPE = "ai"
PLAYER_2_ARCH = "ppo"
MODEL_2_PATH = "models/favourite/PPO/model_v59.pt"

MAX_STEPS = 1000

# --- PLOT INITIALIZATION ---
plt.ion()
fig, ax = plt.subplots(figsize=(8, 4))
(line1,) = ax.plot([], [], "g-", label="Robot 1 (Green)", linewidth=1.5)
(line2,) = ax.plot([], [], "b-", label="Robot 2 (Blue)", linewidth=1.5)
ax.set_xlabel("Step")
ax.set_ylabel("Total Reward")
ax.legend()
ax.grid(True, alpha=0.3)


def load_ai_model(path, arch, device):
    if not os.path.exists(path):
        print(f"⚠️ Model not found: {path}")
        return None

    try:
        if arch == "a2c":
            model = ActorCriticNet(obs_size=11).to(device)
        elif arch == "ppo":
            model = create_agent(11, 128).to(device)
        else:
            return None

        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading {arch.upper()}: {e}")
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
                act, _, _, _ = select_action(model, obs_vec, DEVICE)
                return act.flatten()
            elif arch == "ppo":
                obs_t = torch.FloatTensor(obs_vec).to(DEVICE).unsqueeze(0)
                action_params, _ = model(obs_t)
                mu, _ = torch.chunk(action_params, 2, dim=-1)
                return torch.tanh(mu).cpu().numpy().flatten()
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

    print(f"\n🚀 MATCH START: {PLAYER_1_ARCH.upper()} vs {PLAYER_2_ARCH.upper()}")

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

            # Calculate rewards using A2C reward function for cross-comparison
            r1_s = get_reward(
                None, info, done, state[0], info.get("is_collision", False)
            )
            r2_s = get_reward(
                None, info, done, state[1], info.get("is_collision", False)
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
                ax.set_title(
                    f"Round {round_count+1} | {PLAYER_1_ARCH} vs {PLAYER_2_ARCH}"
                )
                fig.canvas.flush_events()

            sys.stdout.write(
                f"\rStep: {step_count:4d} | R1 ({PLAYER_1_ARCH}): {total_r1:7.1f} | R2 ({PLAYER_2_ARCH}): {total_r2:7.1f}"
            )
            sys.stdout.flush()

            step_count += 1
            env.render()

        round_count += 1
        winner = info.get("winner")
        if winner == 1:
            scores[0] += 1
        elif winner == 2:
            scores[1] += 1

        print(f"\n--- Round {round_count} Over ---")
        print(f"Winner: Robot {winner} | Score: {scores[0]} - {scores[1]}")

        pygame.time.wait(1000)

        # Hot-reload models for live progress monitoring
        if PLAYER_1_TYPE == "ai":
            m1 = load_ai_model(MODEL_1_PATH, PLAYER_1_ARCH, DEVICE)
        if PLAYER_2_TYPE == "ai":
            m2 = load_ai_model(MODEL_2_PATH, PLAYER_2_ARCH, DEVICE)


if __name__ == "__main__":
    main()
