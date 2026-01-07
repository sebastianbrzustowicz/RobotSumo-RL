def get_reward(env, info, done, state_ai, is_collision):
    if done:
        winner = info.get("winner")
        if winner == 1:
            return 50.0
        if winner == 2:
            return -30.0
        return -20.0  # Penalty for a draw (timeout/passive play)

    r = 0.0
    v_fwd = state_ai[0]
    v_side = state_ai[1]
    omega = abs(state_ai[2])
    dist_opp = state_ai[5]
    cos_to_opp = state_ai[7]
    dist_edge_norm = state_ai[8]
    cos_to_center = state_ai[10]

    # Reward forward velocity to encourage movement
    if v_fwd > 0.2:
        r += 0.03 * v_fwd

    if is_collision:
        # Penalty for side-collision (anti-dancing)
        if cos_to_opp < 0.6:
            r -= 0.15
        else:
            # Reward pushing with the front of the robot
            r += 0.5 * v_fwd

            # Bonus for pushing the opponent towards the edge
            if dist_edge_norm < 0.5 and cos_to_opp > 0.9:
                r += 0.1 * (1.0 - dist_edge_norm)
    else:
        # Reward tracking and approaching the opponent
        if cos_to_opp > 0.8:
            r += 0.02 * cos_to_opp

    # Penalty for excessive rotation without forward movement
    if omega > 0.5 and v_fwd < 0.2:
        r -= 0.1 * omega

    # Survival logic: penalty for facing the wrong way near the edge
    if dist_edge_norm < 0.25:
        danger = (0.25 - dist_edge_norm) / 0.25
        if cos_to_center < 0:
            r -= 0.4 * danger

    # Time penalty to encourage faster victories
    r -= 0.03

    return float(r)
