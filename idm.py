def idm_acceleration(v_i, v_lead, x_i, x_lead,
                     v_des=30.0, a_max=3.5, b=1.5, d0=5.0, T=1.7, delta=4.0, L=5.0):
    """
    IDM-based acceleration computation for a following vehicle.
    
    Parameters:
        v_i: current velocity of vehicle i
        v_lead: velocity of the leading vehicle
        x_i: position of vehicle i
        x_lead: position of the leading vehicle
        L: length of the vehicle
        
    Returns:
        a_i: acceleration of vehicle i
    """
    delta_v = v_i - v_lead
    d_i = max(x_lead - x_i - L, 0.1)  # prevent divide-by-zero
    s_star = d0 + v_i * T + (v_i * delta_v) / (2.0 * (a_max * b) ** 0.5)

    a = a_max * (1 - (v_i / v_des) ** delta - (s_star / d_i) ** 2)
    return a
