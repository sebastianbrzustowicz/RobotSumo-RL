import math

def get_reward(env, info, done, state_ai, had_collision):
    if done:
        winner = info.get("winner")
        if winner == 1: return 1000.0
        if winner == 2: return -500.0
        return -50.0

    reward = 0.0
    v_linear = state_ai[0]
    omega = state_ai[2]
    dist_opp_norm = state_ai[5]
    cos_to_opp = state_ai[7]
    dist_edge_norm = state_ai[8]

    if v_linear > 0.5:
        reward += 2.0 * (v_linear ** 2)

    if cos_to_opp > 0.0:
        charge_score = v_linear * (cos_to_opp ** 4)
        if v_linear > 0:
            reward += 8.0 * charge_score
            if v_linear > 0.9 and cos_to_opp > 0.96:
                reward += 5.0

    if info.get('collision', False):
        if cos_to_opp > 0.8:
            reward += 20.0 * (v_linear ** 2)
        else:
            reward += 2.0

    if dist_edge_norm < 0.2:
        reward -= 40.0 * (1.0 - (dist_edge_norm / 0.2))

    if cos_to_opp > 0.0 and v_linear < 0:
        reward -= 5.0 * abs(v_linear)

    v_abs = abs(v_linear)
    omega_abs = abs(omega)
    
    if v_abs < 0.1:
        if omega_abs < 0.1:
            reward -= 5.0
        elif omega_abs > 0.7:
            reward -= 1.0

    reward -= 0.2 
    
    return reward