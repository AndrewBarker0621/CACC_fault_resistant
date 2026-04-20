import numpy as np
import matplotlib.pyplot as plt
from idm import idm_acceleration

DT = 0.1
SIM_TIME = 100.0
T = int(SIM_TIME / DT)
time = np.arange(0, SIM_TIME, DT)
np.random.seed(42)

L = 5.0  # length
gaps = [19, 4.5, 8, 6.5]         # init distance 
speeds = [10.0, 17.5, 16.0, 15.5, 12.0]  # init vel


def get_dynamic_acceleration_with_noise(t, scenario):
    t_sec = t * DT
    noise = np.random.normal(0, 0.1)

    if scenario == 1:
        if t_sec < 20:
            return 0.0 + noise * 0.3
        elif 20 <= t_sec < 50:
            return 0.3 * np.sin(0.1 * np.pi * t_sec) + noise
        else:
            return 0.0 + noise
    elif scenario == 2:
        SC2_START, SC2_END = 30.0, 70.0
        SC2_BURST_LEN = 6.0
        SC2_PLATEAU_LEN = 3.0
        SC2_CYCLE = SC2_BURST_LEN + SC2_PLATEAU_LEN
        SC2_AMPS = [1.6, 0.9, 1.3, 0.7, 1.5]
        SC2_OMEGAS = [0.5*np.pi, 0.35*np.pi, 0.6*np.pi, 0.4*np.pi, 0.55*np.pi]

        if SC2_START <= t_sec < SC2_END:
            phase = t_sec - SC2_START
            cycle_idx = int(phase // SC2_CYCLE)
            in_cycle_time = phase - cycle_idx * SC2_CYCLE
            A = SC2_AMPS[cycle_idx % len(SC2_AMPS)]
            omega = SC2_OMEGAS[cycle_idx % len(SC2_OMEGAS)]

            if in_cycle_time < SC2_BURST_LEN:
                return A * np.sin(omega * in_cycle_time) + noise
            else:
                return 0.0 + noise * 0.3
        else:
            return 0.0 + noise
    elif scenario == 3:
        if 20 <= t_sec < 40:
            return 0.4 + noise
        elif 40 <= t_sec < 45:
            return -3.0 + noise
        elif 45 <= t_sec < 55:
            return 0.0 + noise * 0.3
        elif 55 <= t_sec < 65:
            return 1.0 + noise
        else:
            return 0.0 + noise
    return 0.0 + noise


def run_simulation(scenario):
    num_vehicles = len(speeds)

    # init pos
    positions = [100.0]
    for i in range(1, num_vehicles):
        prev_x = positions[i - 1]
        positions.append(prev_x - gaps[i - 1] - L)
    positions = np.array(positions, dtype=np.float64)

    velocities = np.array(speeds, dtype=np.float64)

    # his
    history_pos = [[] for _ in range(num_vehicles)]
    history_vel = [[] for _ in range(num_vehicles)]
    history_gap = [[] for _ in range(num_vehicles - 1)]

    for t in range(T):
        # leader acc
        a0 = get_dynamic_acceleration_with_noise(t, scenario)
        velocities[0] = max(velocities[0] + a0 * DT, 0.0)
        positions[0] = positions[0] + velocities[0] * DT

        # follower (IDM)
        for i in range(1, num_vehicles):
            vi = velocities[i]
            vi_lead = velocities[i - 1]
            xi = positions[i]
            xi_lead = positions[i - 1]

            acc = idm_acceleration(vi, vi_lead, xi, xi_lead, L=L)
            acc += np.random.normal(0, 0.05)  # noise
            velocities[i] = max(vi + acc * DT, 0.0)
            positions[i] = xi + velocities[i] * DT

        # log
        for i in range(num_vehicles):
            history_pos[i].append(positions[i])
            history_vel[i].append(velocities[i])
        for i in range(1, num_vehicles):
            gap = positions[i - 1] - positions[i] - L
            history_gap[i - 1].append(gap)

    return history_vel, history_gap


# scenario
for scenario in [1, 2, 3]:
    v_histories, gap_histories = run_simulation(scenario)

    # --------- vel ---------
    plt.figure(figsize=(10, 5))
    for i, v_list in enumerate(v_histories):
        plt.plot(time, v_list, label=f'Vehicle {i}')
    plt.ylabel("Velocity (m/s)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"scenario_{scenario}_velocity.png", dpi=300)
    plt.close()

    # --------- gap ---------
    plt.figure(figsize=(10, 5))
    for i, g_list in enumerate(gap_histories):
        plt.plot(time, g_list, label=f'Gap {i}-{i+1}')
    plt.ylabel("Gap (m)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"scenario_{scenario}_gap.png", dpi=300)
    plt.close()

