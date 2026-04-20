import matplotlib.pyplot as plt
import numpy as np

# sim para
DT = 0.1
T = int(100 / DT)
time = np.arange(0, T * DT, DT)

# random seed
np.random.seed(42)

# Sce 2
SC2_START, SC2_END = 30.0, 70.0
SC2_BURST_LEN = 6.0     
SC2_PLATEAU_LEN = 3.0   
SC2_CYCLE = SC2_BURST_LEN + SC2_PLATEAU_LEN

SC2_AMPS = [1.6, 0.9, 1.3, 0.7, 1.5]                 
SC2_OMEGAS = [0.5*np.pi, 0.35*np.pi, 0.6*np.pi, 0.4*np.pi, 0.55*np.pi]  

def get_dynamic_acceleration_with_noise(t, scenario):
    t_sec = t * DT
    noise = np.random.normal(0, 0.1)  # σ=0.15

    if scenario == 1:
        if t_sec < 20:
            return 0.0 + noise * 0.3   # constant
        elif 20 <= t_sec < 50:
            return 0.3 * np.sin(0.1 * np.pi * t_sec) + noise
        else:
            return 0.0 + noise

    elif scenario == 2:
        if SC2_START <= t_sec < SC2_END:
            phase = t_sec - SC2_START
            cycle_idx = int(phase // SC2_CYCLE)
            in_cycle_time = phase - cycle_idx * SC2_CYCLE

            A = SC2_AMPS[cycle_idx % len(SC2_AMPS)]
            omega = SC2_OMEGAS[cycle_idx % len(SC2_OMEGAS)]

            if in_cycle_time < SC2_BURST_LEN:
                # sin
                return A * np.sin(omega * in_cycle_time) + noise
            else:
                # constant
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

# init
initial_speed = 10.0
velocities = {1: [], 2: [], 3: []}
accelerations = {1: [], 2: [], 3: []}


def simulate_scenario(scenario):
    v = 10.0
    v_list = []
    a_list = []
    for t in range(T):
        a = get_dynamic_acceleration_with_noise(t, scenario)
        v = max(v + a * DT, 0.0)
        v_list.append(v)
        a_list.append(a)
    return np.array(v_list), np.array(a_list)

for scenario in [1, 2, 3]:
    v, a = simulate_scenario(scenario)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Twin axis for acceleration (draw first, so it's under the velocity curve)
    ax2 = ax1.twinx()
    ax2.plot(time, a, color='tab:blue', linestyle="--", alpha=0.5, label="Acceleration (m/s$^2$)")
    ax2.set_ylabel("Acceleration (m/s$^2$)", color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylim(-3.5, 2)

    # Velocity in red (draw last so it is on top)
    ax1.plot(time, v, color='tab:red', linewidth=2, label="Velocity (m/s)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Velocity (m/s)", color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_ylim(0, 20)
    ax1.grid(True)

    # Combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")

    fname = f"scenario_{scenario}_leader_vel_acc.png"
    fig.tight_layout()
    fig.savefig(f"./{fname}", dpi=600)
    plt.close(fig)


# for scenario in [1, 2, 3]:
#     v = initial_speed
#     v_list = []
#     a_list = []
#     for t in range(T):
#         a = get_dynamic_acceleration_with_noise(t, scenario)
#         v += a * DT
#         v = max(v, 0.0)
#         v_list.append(v)
#         a_list.append(a)
#     velocities[scenario] = v_list
#     accelerations[scenario] = a_list

# fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# for scenario in [1, 2, 3]:
#     axs[0].plot(time, velocities[scenario], label=f'Scenario {scenario}')
# axs[0].set_ylabel("Velocity (m/s)")
# axs[0].set_title("Leader Vehicle Velocity with Noisy Acceleration")
# axs[0].grid(True)
# axs[0].legend()

# for scenario in [1, 2, 3]:
#     axs[1].plot(time, accelerations[scenario], label=f'Scenario {scenario}')
# axs[1].set_xlabel("Time (s)")
# axs[1].set_ylabel("Acceleration (m/s²)")
# axs[1].set_title("Leader Vehicle Acceleration with Noise")
# axs[1].grid(True)
# axs[1].legend()

# plt.tight_layout()
# plt.savefig("vel_acc.png")
# plt.show()
