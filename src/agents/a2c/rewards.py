def get_reward(env, info, done, state_ai, is_collision):
    # --- 1. TERMINAL REWARDS ---
    if done:
        winner = info.get("winner")
        if winner == 1: 
            return 4000.0   
        if winner == 2: 
            return -1000.0  
        return -500.0       

    reward = 0.0
    
    # Input parameters
    v_linear = state_ai[0]       
    omega = abs(state_ai[2]) 
    cos_to_opp = state_ai[7]
    dist_edge_norm = state_ai[8]
    cos_to_center = state_ai[9] 

    is_facing_opponent = cos_to_opp > 0.90 
    is_perfect_aim = cos_to_opp > 0.97

    # --- 2. BACKWARD DRIVING BLOCK ---
    if v_linear < 0:
        # Return immediately and exit the function here
        return -20.0 * abs(v_linear)

    # --- 3. ANTI-SPINNING ---
    if omega > 0.4:
        reward -= 8.0 * (omega ** 2)

    # --- 4. FORWARD MOVEMENT REWARDING ---
    drive_quality = max(0, cos_to_opp)
    
    if v_linear > 0:
        reward += 0.2 * v_linear 
        reward += 5.0 * v_linear * drive_quality

    # --- 5. NITRO / CHARGE ---
    if v_linear > 0.6 and is_facing_opponent:
        reward += 100.0 * (v_linear ** 2)
        if is_perfect_aim:
            reward += 15.0 * v_linear

    # --- 6. EDGE LOGIC ---
    if dist_edge_norm < 0.35:
        danger_scale = (0.35 - dist_edge_norm) / 0.35
        if cos_to_center < 0: 
            reward -= 40.0 * v_linear * danger_scale
        elif cos_to_center > 0.5: 
            reward += 5.0 * v_linear * danger_scale

    # --- 7. COMBAT DYNAMICS (COLLISIONS) ---
    if is_collision:
        if is_facing_opponent:
            reward += 50.0 
            reward += 100.0 * v_linear
        else:
            reward -= 20.0 

    # --- 8. TIME COST ---
    reward -= 1.0 

    return float(reward)