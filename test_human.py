import pygame
import torch
import numpy as np
import os
import sys
from env.sumo_env import SumoEnv
from model import ActorCriticNet, select_action
from env.config import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PLAYER CONFIGURATION ---
PLAYER_1_TYPE = "human" # "human" / "ai" / "dummy"
MODEL_1_PATH = "models/chosen/model_v101.pt"

PLAYER_2_TYPE = "ai" # "human" / "ai" / "dummy"
MODEL_2_PATH = "models/sumo_push_master.pt"

def load_ai_model(path, device, debug=True):
    if not os.path.exists(path):
        print(f"⚠️ Model nie znaleziony: {path}")
        return None
    
    model = ActorCriticNet(obs_size=11).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    if debug:
        print(f"✅ Model loaded: {path}")
    return model

def get_action(p_type, robot_idx, state, model):
    if p_type == "dummy":
        return [0.0, 0.0]
    
    if p_type == "human":
        keys = pygame.key.get_pressed()
        v, omega = 0.0, 0.0
        if robot_idx == 0:
            if keys[pygame.K_UP]:    v = 1.0
            if keys[pygame.K_DOWN]:  v = -1.0
            if keys[pygame.K_LEFT]:  omega = 1.0
            if keys[pygame.K_RIGHT]: omega = -1.0
        else:
            if keys[pygame.K_w]:     v = 1.0
            if keys[pygame.K_s]:     v = -1.0
            if keys[pygame.K_a]:     omega = 1.0
            if keys[pygame.K_d]:     omega = -1.0
        return [v, omega]
    
    if p_type == "ai":
        if model is None: return [0.0, 0.0]
        obs = state[robot_idx] 
        with torch.no_grad():
            act_np, _, _, _ = select_action(model, obs, DEVICE)
            return act_np.flatten()

def main():
    env = SumoEnv(render_mode=True, render_vectors=True)
    
    model1 = load_ai_model(MODEL_1_PATH, DEVICE) if PLAYER_1_TYPE == "ai" else None
    model2 = load_ai_model(MODEL_2_PATH, DEVICE) if PLAYER_2_TYPE == "ai" else None

    scores = [0, 0]
    round_count = 0
    running = True
    state = env.reset(randPositions=False)

    print("\n--- START OF THE FIGHT ---")
    print(f"Robot 1: {PLAYER_1_TYPE} ({os.path.basename(MODEL_1_PATH) if model1 else 'Manual'})")
    print(f"Robot 2: {PLAYER_2_TYPE} ({os.path.basename(MODEL_2_PATH) if model2 else 'Manual'})")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        act1 = get_action(PLAYER_1_TYPE, 0, state, model1)
        act2 = get_action(PLAYER_2_TYPE, 1, state, model2)

        state, _, done, info = env.step(act1, act2)
        env.render()

        if done:
            round_count += 1
            winner = info.get("winner")
            
            if winner == 1:
                scores[0] += 1
                name = os.path.basename(MODEL_1_PATH) if PLAYER_1_TYPE == "ai" else "Human 1"
                result_text = f"Winner: Robot 1 [{name}]"
            elif winner == 2:
                scores[1] += 1
                name = os.path.basename(MODEL_2_PATH) if PLAYER_2_TYPE == "ai" else "Human 2"
                result_text = f"Winner: Robot 2 [{name}]"
            else:
                result_text = "DRAW!"

            print(f"\nRunda {round_count}: {result_text}")
            print(f"SCOREBOARD:")
            print(f"  R1: {scores[0]} pkt")
            print(f"  R2: {scores[1]} pkt")
            print("-" * 30)

            pygame.time.wait(1000)
            state = env.reset(randPositions=True)
            model1 = load_ai_model(MODEL_1_PATH, DEVICE, debug=False) if PLAYER_1_TYPE == "ai" else None
            model2 = load_ai_model(MODEL_2_PATH, DEVICE, debug=False) if PLAYER_2_TYPE == "ai" else None

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()