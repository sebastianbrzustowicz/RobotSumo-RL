import os
import sys

import matplotlib.pyplot as plt
import pygame
import torch

from src.agents.PPO.agent import create_agent
from src.agents.PPO.rewards import get_reward
from src.env.sumo_env import SumoEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PLAYER_1_TYPE = "ai"
MODEL_1_PATH = "models/favourite/PPO/model_v59.pt"
PLAYER_2_TYPE = "ai"
MODEL_2_PATH = "models/favourite/PPO/model_v59.pt"

MAX_STEPS = 1000

plt.ion()
fig, ax = plt.subplots(figsize=(8, 4))
(line1,) = ax.plot([], [], "g-", label="Robot 1 (Green)", linewidth=1.5)
(line2,) = ax.plot([], [], "b-", label="Robot 2 (Blue)", linewidth=1.5)
fig.canvas.manager.set_window_title('Live Reward Analytics')
ax.set_title("Cumulative Reward")
ax.set_xlabel("Step")
ax.set_ylabel("Total Reward")
ax.legend()
ax.grid(True, alpha=0.3)


def load_ai_model(path, device):
    if not os.path.exists(path):
        return None
    model = create_agent(11, 128).to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
    except:
        return None


def get_action(p_type, robot_idx, state, model):
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
        obs = torch.FloatTensor(state[robot_idx]).to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            action_params, _ = model(obs)
            mu, _ = torch.chunk(action_params, 2, dim=-1)
            return torch.tanh(mu).cpu().numpy().flatten()
    return [0.0, 0.0]


def main():
    env = SumoEnv(render_mode=True)
    model1 = load_ai_model(MODEL_1_PATH, DEVICE) if PLAYER_1_TYPE == "ai" else None
    model2 = load_ai_model(MODEL_2_PATH, DEVICE) if PLAYER_2_TYPE == "ai" else None

    scores = [0, 0]
    round_count = 0

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
                    return

            act1 = get_action(PLAYER_1_TYPE, 0, state, model1)
            act2 = get_action(PLAYER_2_TYPE, 1, state, model2)

            state, _, env_done, info = env.step(act1, act2)
            if isinstance(state, tuple):
                state = state[0]

            done = env_done or (step_count + 1 >= MAX_STEPS)

            # --- Adjust win/lose info for robot 2 perspective ---
            actual_winner = info.get("winner")
            info_for_robot2 = info.copy()

            if actual_winner == 1:
                info_for_robot2["winner"] = 2
            elif actual_winner == 2:
                info_for_robot2["winner"] = 1
            else:
                info_for_robot2["winner"] = actual_winner
            # ------------------------------------

            r1_s = get_reward(
                None, info, done, state[0], info.get("is_collision", False)
            )
            r2_s = get_reward(
                None, info_for_robot2, done, state[1], info.get("is_collision", False)
            )

            total_r1 += r1_s
            total_r2 += r2_s

            steps_h.append(step_count)
            r1_h.append(total_r1)
            r2_h.append(total_r2)

            if step_count % 5 == 0:
                line1.set_data(steps_h, r1_h)
                line2.set_data(steps_h, r2_h)
                ax.relim()
                ax.autoscale_view()
                # ax.set_title(f"Round {round_count+1} | Score: {scores[0]}-{scores[1]}")
                ax.set_title(f"Cumulative Reward")
                fig.canvas.flush_events()

            sys.stdout.write(
                f"\rStep: {step_count:4d} | R1: {total_r1:7.1f} | R2: {total_r2:7.1f}"
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

        print(f"\nRound {round_count} Over. Winner: {winner}")

        pygame.time.wait(2000)

        if PLAYER_1_TYPE == "ai":
            model1 = load_ai_model(MODEL_1_PATH, DEVICE)
        if PLAYER_2_TYPE == "ai":
            model2 = load_ai_model(MODEL_2_PATH, DEVICE)


if __name__ == "__main__":
    main()
