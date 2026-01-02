import os
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from env.sumo_env import SumoEnv
from model import ActorCriticNet, select_action

# --- CONFIGURATION ---
MODELS_DIR = "models/chosen/"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = "models/results/"
RESULTS_PATH = os.path.join(RESULTS_DIR, f"elo_rankings_{TIMESTAMP}.txt")

N_FIGHTS = 50           # Number of matches played between every unique pair of models
BASE_ELO = 1200         # Starting rating assigned to all models before the tournament
K_FACTOR = 32           # Sensitivity constant: determines how much a rating changes after each match
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_ai_model(path, device):
    model = ActorCriticNet(obs_size=11).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def calculate_expected(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

def update_elo(ra, rb, score_a):
    ea = calculate_expected(ra, rb)
    eb = 1 - ea
    new_ra = ra + K_FACTOR * (score_a - ea)
    new_rb = rb + K_FACTOR * ((1 - score_a) - eb)
    return new_ra, new_rb

def main():
    # Ensure the results directory exists
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]
    if len(model_files) < 2:
        print("❌ At least two models are required for ranking.")
        return

    elo_ratings = {name: BASE_ELO for name in model_files}
    loaded_models = {name: load_ai_model(os.path.join(MODELS_DIR, name), DEVICE) for name in model_files}
    
    env = SumoEnv(render_mode=False)
    
    print(f"🚀 Starting Tournament: {len(model_files)} models, {N_FIGHTS} rounds per pair.")

    pairs = []
    for i in range(len(model_files)):
        for j in range(i + 1, len(model_files)):
            pairs.append((model_files[i], model_files[j]))

    total_matches = len(pairs) * N_FIGHTS
    
    with tqdm(total=total_matches, desc="Tournament Progress") as pbar:
        for name_a, name_b in pairs:
            model_a = loaded_models[name_a]
            model_b = loaded_models[name_b]
            
            for _ in range(N_FIGHTS):
                state = env.reset(randPositions=True)
                done = False
                
                while not done:
                    with torch.no_grad():
                        act1, _, _, _ = select_action(model_a, state[0], DEVICE)
                        act1 = act1.flatten()
                        
                        act2, _, _, _ = select_action(model_b, state[1], DEVICE)
                        act2 = act2.flatten()
                    
                    state, _, done, info = env.step(act1, act2)
                
                winner = info.get("winner")
                if winner == 1:
                    score = 1.0  
                elif winner == 2:
                    score = 0.0  
                else:
                    score = 0.5  

                new_a, new_b = update_elo(elo_ratings[name_a], elo_ratings[name_b], score)
                elo_ratings[name_a] = new_a
                elo_ratings[name_b] = new_b
                pbar.update(1)

    sorted_ranking = sorted(elo_ratings.items(), key=lambda item: item[1], reverse=True)
    
    header = "🏆 FINAL ELO RANKINGS:"
    print(f"\n{header}")
    
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(f"{header}\n")
        for i, (name, score) in enumerate(sorted_ranking):
            line = f"{i+1}. {name:25} | ELO: {int(score)}"
            print(line)
            f.write(f"{line}\n")
    
    print(f"\n✅ Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()