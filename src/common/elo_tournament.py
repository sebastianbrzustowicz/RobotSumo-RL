import os
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from src.agents.A2C.networks import ActorCriticNet, select_action
from src.agents.PPO.agent import create_agent
from src.common.tournament_plots import save_tournament_plots
from src.env.sumo_env import SumoEnv

# --- CONFIGURATION ---
A2C_DIR = "models/favourite/A2C/"
PPO_DIR = "models/favourite/PPO/"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_TOURNAMENT_DIR = os.path.join(
    "results", "tournaments", f"tournament_elo_{TIMESTAMP}"
)
LEADERBOARD_PATH = os.path.join(CURRENT_TOURNAMENT_DIR, "leaderboard.txt")

N_FIGHTS = 50
BASE_ELO = 1200
K_FACTOR = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 1000


def load_ai_model(path, arch, device):
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
        print(f"Load Error {path}: {e}")
        return None


def get_tournament_action(model, arch, obs, device):
    with torch.no_grad():
        if arch == "a2c":
            act, _, _, _ = select_action(model, obs, device)
            return act.flatten()
        elif arch == "ppo":
            obs_t = torch.FloatTensor(obs).to(device).unsqueeze(0)
            action_params, _ = model(obs_t)
            mu, _ = torch.chunk(action_params, 2, dim=-1)
            return torch.tanh(mu).cpu().numpy().flatten()
    return np.zeros(2)


def calculate_expected(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def update_elo(ra, rb, score_a):
    ea = calculate_expected(ra, rb)
    new_ra = ra + K_FACTOR * (score_a - ea)
    new_rb = rb + K_FACTOR * ((1 - score_a) - (1 - ea))
    return new_ra, new_rb


def main():
    os.makedirs(CURRENT_TOURNAMENT_DIR, exist_ok=True)

    models_metadata = []

    if os.path.exists(A2C_DIR):
        for f in os.listdir(A2C_DIR):
            if f.endswith(".pt"):
                models_metadata.append(
                    {
                        "name": f,
                        "path": os.path.join(A2C_DIR, f),
                        "arch": "A2C",
                    }
                )

    if os.path.exists(PPO_DIR):
        for f in os.listdir(PPO_DIR):
            if f.endswith(".pt"):
                models_metadata.append(
                    {
                        "name": f,
                        "path": os.path.join(PPO_DIR, f),
                        "arch": "PPO",
                    }
                )

    if len(models_metadata) < 2:
        print("Not enough models to start a tournament.")
        return

    print(f"Loading {len(models_metadata)} models on {DEVICE}...")
    loaded_models = {}
    elo_ratings = {}

    for m in models_metadata:
        model_obj = load_ai_model(m["path"], m["arch"].lower(), DEVICE)
        if model_obj:
            model_id = f"{m['arch']}_{m['name']}"
            loaded_models[model_id] = {
                "model": model_obj,
                "arch": m["arch"].lower(),
                "display_name": m["name"],
                "type": m["arch"],
            }
            elo_ratings[model_id] = BASE_ELO

    model_ids = list(loaded_models.keys())
    env = SumoEnv(render_mode=False)

    pairs = []
    for i in range(len(model_ids)):
        for j in range(i + 1, len(model_ids)):
            pairs.append((model_ids[i], model_ids[j]))

    total_matches = len(pairs) * N_FIGHTS
    print(f"Tournament Start: {len(model_ids)} models, {total_matches} total matches.")

    with tqdm(total=total_matches, desc="Tournament Progress") as pbar:
        for id_a, id_b in pairs:
            m_a_data = loaded_models[id_a]
            m_b_data = loaded_models[id_b]

            for _ in range(N_FIGHTS):
                state = env.reset(randPositions=True)
                if isinstance(state, tuple):
                    state = state[0]
                done = False
                steps = 0
                while not done and steps < MAX_STEPS:
                    act_a = get_tournament_action(
                        m_a_data["model"], m_a_data["arch"], state[0], DEVICE
                    )
                    act_b = get_tournament_action(
                        m_b_data["model"], m_b_data["arch"], state[1], DEVICE
                    )
                    state, _, done, info = env.step(act_a, act_b)
                    if isinstance(state, tuple):
                        state = state[0]
                    steps += 1

                winner = info.get("winner")
                score = 1.0 if winner == 1 else (0.0 if winner == 2 else 0.5)
                elo_ratings[id_a], elo_ratings[id_b] = update_elo(
                    elo_ratings[id_a], elo_ratings[id_b], score
                )
                pbar.update(1)

    sorted_ranking = sorted(elo_ratings.items(), key=lambda item: item[1], reverse=True)

    header_fmt = "{:<5} | {:<10} | {:<25} | {:<5}"
    row_fmt = "{:<5} | {:<10} | {:<25} | {:<5}"

    output = []
    output.append("=" * 60)
    output.append(f"🏆 TOURNAMENT RESULTS - {TIMESTAMP}")
    output.append(f"Fights per pair: {N_FIGHTS} | K-Factor: {K_FACTOR}")
    output.append("=" * 60)
    output.append(header_fmt.format("Rank", "Agent", "Model File", "ELO"))
    output.append("-" * 60)

    for i, (m_id, score) in enumerate(sorted_ranking):
        m_info = loaded_models[m_id]
        output.append(
            row_fmt.format(i + 1, m_info["type"], m_info["display_name"], int(score))
        )

    result_text = "\n".join(output)
    print("\n" + result_text)

    with open(LEADERBOARD_PATH, "w", encoding="utf-8") as f:
        f.write(result_text)

    save_tournament_plots(
        sorted_ranking, loaded_models, CURRENT_TOURNAMENT_DIR, TIMESTAMP
    )

    print(f"\nResults and plots saved to: {CURRENT_TOURNAMENT_DIR}")


if __name__ == "__main__":
    main()
