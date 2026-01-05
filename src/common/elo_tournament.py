import os
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from src.env.sumo_env import SumoEnv
from src.agents.A2C.networks import ActorCriticNet, select_action
from src.agents.PPO.agent import create_agent

# --- CONFIGURATION ---
A2C_DIR = "models/favourite/A2C/"
PPO_DIR = "models/favourite/PPO/"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = "results/"
RESULTS_PATH = os.path.join(RESULTS_DIR, f"tournament_elo_{TIMESTAMP}.txt")

N_FIGHTS = 50
BASE_ELO = 1200         
K_FACTOR = 32           
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 1000

def load_ai_model(path, arch, device):
    """Loads model weights based on architecture type."""
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
    """Retrieves action based on the specific agent architecture."""
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
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Collect models from directories
    models_metadata = [] 
    
    if os.path.exists(A2C_DIR):
        for f in os.listdir(A2C_DIR):
            if f.endswith(".pt"):
                models_metadata.append({"name": f"A2C_{f}", "path": os.path.join(A2C_DIR, f), "arch": "a2c"})
                
    if os.path.exists(PPO_DIR):
        for f in os.listdir(PPO_DIR):
            if f.endswith(".pt"):
                models_metadata.append({"name": f"PPO_{f}", "path": os.path.join(PPO_DIR, f), "arch": "ppo"})

    if len(models_metadata) < 2:
        print("Not enough models to start a tournament.")
        return

    # 2. Initialize ratings and load models
    print(f"Loading {len(models_metadata)} models on {DEVICE}...")
    loaded_models = {}
    elo_ratings = {}
    
    for m in models_metadata:
        model_obj = load_ai_model(m["path"], m["arch"], DEVICE)
        if model_obj:
            loaded_models[m["name"]] = {"model": model_obj, "arch": m["arch"]}
            elo_ratings[m["name"]] = BASE_ELO

    model_names = list(loaded_models.keys())
    env = SumoEnv(render_mode=False)
    
    # 3. Create round-robin pairs
    pairs = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            pairs.append((model_names[i], model_names[j]))

    total_matches = len(pairs) * N_FIGHTS
    print(f"Tournament Start: {len(model_names)} models, {total_matches} total matches.")

    # 4. Tournament Loop
    with tqdm(total=total_matches, desc="Tournament Progress") as pbar:
        for name_a, name_b in pairs:
            m_a_data = loaded_models[name_a]
            m_b_data = loaded_models[name_b]
            
            for _ in range(N_FIGHTS):
                state = env.reset(randPositions=True)
                if isinstance(state, tuple): state = state[0]
                done = False
                steps = 0
                
                while not done and steps < MAX_STEPS:
                    act_a = get_tournament_action(m_a_data["model"], m_a_data["arch"], state[0], DEVICE)
                    act_b = get_tournament_action(m_b_data["model"], m_b_data["arch"], state[1], DEVICE)
                    
                    state, _, done, info = env.step(act_a, act_b)
                    if isinstance(state, tuple): state = state[0]
                    steps += 1
                
                winner = info.get("winner")
                score = 1.0 if winner == 1 else (0.0 if winner == 2 else 0.5)

                elo_ratings[name_a], elo_ratings[name_b] = update_elo(
                    elo_ratings[name_a], elo_ratings[name_b], score
                )
                pbar.update(1)

    # 5. Process Results
    sorted_ranking = sorted(elo_ratings.items(), key=lambda item: item[1], reverse=True)
    
    output = []
    output.append("="*60)
    output.append(f"🏆 TOURNAMENT RESULTS - {TIMESTAMP}")
    output.append(f"Fights per pair: {N_FIGHTS} | K-Factor: {K_FACTOR}")
    output.append("="*60)
    output.append(f"{'Rank':<5} | {'Model Name':<35} | {'ELO'}")
    output.append("-" * 60)

    for i, (name, score) in enumerate(sorted_ranking):
        output.append(f"{i+1:<5} | {name:<35} | {int(score)}")

    result_text = "\n".join(output)
    print("\n" + result_text)
    
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(result_text)
    
    print(f"\n✅ Results saved to: {RESULTS_PATH}")

if __name__ == "__main__":
    main()