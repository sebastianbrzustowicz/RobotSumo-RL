def get_reward(env, info, done, state_ai, is_collision):
    # --- 1. TERMINAL REWARDS ---
    if done:
        winner = info.get("winner")
        if winner == 1: return 20.0
        if winner == 2: return -20.0
        return -5.0

    r = 0.0
    
    # Input params
    v_linear = state_ai[0]       
    omega = abs(state_ai[2]) 
    cos_to_opp = state_ai[7]
    dist_edge_norm = state_ai[8]
    cos_to_center = state_ai[9] 

    is_facing_opponent = cos_to_opp > 0.90 
    is_perfect_aim = cos_to_opp > 0.97

    # --- 2. BACKWARD DRIVING BLOCK ---
    if v_linear < 0:
        r -= 2.0 * abs(v_linear)

    # --- 3. ANTI-SPINNING (Limiting spinning in circles) ---
    if omega > 0.4:
        r -= 0.5 * (omega ** 2)

    # --- 4. FORWARD MOVEMENT (Reward for driving forward toward the enemy) ---
    drive_quality = max(0, cos_to_opp)
    if v_linear > 0:
        r += 0.1 * v_linear 
        r += 0.5 * v_linear * drive_quality

    # --- 5. CHARGE (Bonus for aggressive attack) ---
    if v_linear > 0.6 and is_facing_opponent:
        r += 2.0 * (v_linear ** 2)
        if is_perfect_aim:
            r += 1.0 * v_linear

    # --- 6. EDGE LOGIC (Edge avoidance) ---
    if dist_edge_norm < 0.35:
        danger_scale = (0.35 - dist_edge_norm) / 0.35
        if cos_to_center < 0: 
            r -= 4.0 * v_linear * danger_scale
        elif cos_to_center > 0.5: 
            r += 0.5 * v_linear * danger_scale

    # --- 7. COMBAT (Collisions) ---
    if is_collision:
        if is_facing_opponent:
            r += 5.0
            r += 10.0 * v_linear
        else:
            r -= 2.0

    # --- 8. TIME COST ---
    r -= 0.05

    return float(r)