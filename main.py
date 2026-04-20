# main.py — CACC with dual KFs + predecessor perfect-comm + KF fusion
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from init import KinematicBicycleModel, PlatoonInitializer
from mpc import CasadiMPCController
from kf import KalmanFilter1D


# -----------------------------
# Leader acceleration profiles
# -----------------------------
SC2_START, SC2_END = 30.0, 70.0
SC2_BURST_LEN = 6.0
SC2_PLATEAU_LEN = 3.0
SC2_CYCLE = SC2_BURST_LEN + SC2_PLATEAU_LEN
SC2_AMPS = [1.6, 0.9, 1.3, 0.7, 1.5]
SC2_OMEGAS = [0.5*np.pi, 0.35*np.pi, 0.6*np.pi, 0.4*np.pi, 0.55*np.pi]


def get_dynamic_acceleration_with_noise(t: int, scenario: int, dt: float = 0.1):
    """Generate leader’s longitudinal acceleration depending on scenario (returns a, delta)."""
    t_sec = t * dt
    noise = np.random.normal(0, 0.1)

    if scenario == 1:
        # Mild sinusoidal acceleration between 20s–50s
        if 20 <= t_sec < 50:
            a = 0.3 * np.sin(0.1 * np.pi * t_sec) + noise * 0.3
        else:
            a = 0.0 + noise * 0.3

    elif scenario == 2:
        # Repeated bursts/plateaus between 30s–70s
        if SC2_START <= t_sec < SC2_END:
            cycle_time = t_sec - SC2_START
            cycle_phase = cycle_time % SC2_CYCLE
            if   cycle_phase < 3:  a = 1.2 + noise
            elif cycle_phase < 6:  a = -2.5 + noise
            elif cycle_phase < 8:  a = 0.0 + noise
            elif cycle_phase < 10: a = 3.0 + noise
            elif cycle_phase < 12: a = 1.5 + noise
            elif cycle_phase < 15: a = -2.8 + noise
            elif cycle_phase < 18: a = 1.0 + noise
            else:                  a = 0.0 + noise
        else:
            a = 0.0 + noise

    elif scenario == 3:
        # Hard brake → cruise → acceleration
        if   20 <= t_sec < 25: a = -3.2 + noise
        elif 25 <= t_sec < 40: a = 0.0
        elif 40 <= t_sec < 55: a = 1.0 + noise
        else:                  a = 0.0 + noise

    elif scenario == 4:
        # Mild accel → hard brake → cruise → accel
        if   20 <= t_sec < 35: a = 0.5 + noise
        elif 35 <= t_sec < 40: a = -3.2 + noise
        elif 40 <= t_sec < 55: a = 0.0 + noise
        elif 55 <= t_sec < 70: a = 1.0 + noise
        else:                  a = 0.0 + noise
    else:
        raise ValueError("Invalid scenario number: 1/2/3/4")

    return a, 0.0


def in_conn_loss(t_sec: float, windows):
    """Return True if current time falls in any outage window [s, e)."""
    for s, e in windows:
        if s <= t_sec < e:
            return True
    return False


def profile_to_flags(profile: str):
    """
    Map ablation profile to fault toggles.

    - noise/bias (including noise_bias/all) only affect predecessor’s local sensor channel.
    - Predecessor communication channel is assumed perfect (no faults).
    - Leader channel only affected by communication faults.
    """
    profile = profile.lower().strip()
    flags = {
        "ENABLE_DELAY": False,         # leader comm delay
        "ENABLE_PACKET_LOSS": False,   # leader packet loss
        "ENABLE_CONN_LOSS": False,     # leader outage
        "ENABLE_MEAS_NOISE": False,    # predecessor sensor noise
        "ENABLE_MEAS_BIAS": False,     # predecessor sensor bias
        "NAME": profile
    }
    if profile == "none":
        pass
    elif profile == "delay":
        flags["ENABLE_DELAY"] = True
    elif profile == "packet_loss":
        flags["ENABLE_PACKET_LOSS"] = True
    elif profile == "conn_loss":
        flags["ENABLE_CONN_LOSS"] = True
    elif profile == "noise":
        flags["ENABLE_MEAS_NOISE"] = True
    elif profile == "bias":
        flags["ENABLE_MEAS_BIAS"] = True
    elif profile == "noise_bias":
        flags["ENABLE_MEAS_NOISE"] = True
        flags["ENABLE_MEAS_BIAS"] = True
    elif profile == "all":
        flags = {k: True for k in flags}
        flags["NAME"] = "all"
    else:
        raise ValueError(f"Unknown ablation profile: {profile}")
    return flags


if __name__ == "__main__":
    # -----------------------------
    # Simulation settings
    # -----------------------------
    np.random.seed(42)

    try:
        scenario = int(input("Enter leader scenario (1/2/3/4): ").strip())
        assert scenario in (1, 2, 3, 4)
    except Exception:
        raise ValueError("Please enter a valid scenario number: 1, 2, 3, or 4.")

    # Fault profile
    profile_name = "none"
    flags = profile_to_flags(profile_name)
    print("Ablation profile:", flags)

    # CACC headway parameters
    D0 = 5.0        # standstill distance [m]
    H_HEADWAY = 1.7 # time headway [s]

    NUM_VEHICLES = 5
    DT = 0.1
    V0 = 10.0
    T = int(100 / DT)

    # --- Leader comm fault parameters ---
    DELAY_SEC = 0.3
    DELAY_STEP = int(DELAY_SEC / DT)
    PACKET_LOSS_RATE = 0.2
    CONN_LOSS_WINDOWS = [(50.0, 60.0)]

    # --- Predecessor sensor fault parameters ---
    POS_NOISE_STD = 0.5
    VEL_NOISE_STD = 0.2
    POS_BIAS = 0.3
    VEL_BIAS = -0.1

    # --- Predecessor perfect-comm + KF fusion weights ---
    # Sequential update: first sensor, then comm (comm has higher confidence)
    R_MEAS = np.diag([max(POS_NOISE_STD**2, 1e-6), max(VEL_NOISE_STD**2, 1e-6)])
    R_COMM = np.diag([1e-4, 1e-4])  # very small to emphasize comm reliability

    # -----------------------------
    # Object initialization
    # -----------------------------
    model = KinematicBicycleModel(wheelbase=2.5, dt=DT)
    mpc = CasadiMPCController(horizon=10, dt=DT, wheelbase=2.5)

    initializer = PlatoonInitializer(num_vehicles=NUM_VEHICLES,
                                     spacing=D0 + H_HEADWAY * V0,
                                     initial_speed=V0)
    gaps   = [19, 4.5, 8, 6.5]       # follower initial gaps
    speeds = [17.5, 16, 15.5, 12.0]  # follower initial speeds
    states = initializer.initialize_custom(
        leader_x=100.0,
        leader_v=10.0,
        follower_gaps=gaps,
        follower_speeds=speeds
    )

    # Leader history (for delayed comm emulation)
    leader_history = []

    # Dual KFs (leader KF + predecessor KF for each follower)
    kf_leader_dict = {i: KalmanFilter1D(dt=DT) for i in range(1, NUM_VEHICLES)}
    kf_pred_dict   = {i: KalmanFilter1D(dt=DT) for i in range(1, NUM_VEHICLES)}
    for i in range(1, NUM_VEHICLES):
        kf_leader_dict[i].initialize(states[0][0],   states[0][3])
        kf_pred_dict[i].initialize(states[i-1][0],   states[i-1][3])

    # Logs
    trajectories = [[] for _ in range(NUM_VEHICLES)]
    state_log = []

    # -----------------------------
    # Simulation loop
    # -----------------------------
    last_u = np.zeros(NUM_VEHICLES)
    for t in range(T):
        print(f"Timestep {t}/{T}")
        t_sec = t * DT
        new_states = []

        for i in range(NUM_VEHICLES):
            delay_active = False
            outage = False
            recv_ok = True

            # True leader at current step
            xL_true_now = states[0][0]
            vL_true_now = states[0][3]
            xL_true_del = np.nan
            vL_true_del = np.nan

            # KF estimates
            xL_hat = np.nan; vL_hat = np.nan
            xP_hat = np.nan; vP_hat = np.nan

            # Predecessor truth / observations (for logging)
            xP_true = np.nan; vP_true = np.nan
            obs_xp  = np.nan; obs_vp  = np.nan     # sensor obs
            xP_comm = np.nan; vP_comm = np.nan     # perfect comm obs
            applied_bias_pred  = False
            applied_noise_pred = False

            if i == 0:
                # Leader dynamics
                a, delta = get_dynamic_acceleration_with_noise(t, scenario, DT)
                leader_history.append(states[0])
            else:
                # ---------- Leader channel (comm faults only) ----------
                if flags["ENABLE_DELAY"] and (t - DELAY_STEP >= 0):
                    delay_active = True
                    xL_true_del, _, _, vL_true_del = leader_history[t - DELAY_STEP]
                else:
                    xL_true_del, vL_true_del = xL_true_now, vL_true_now

                outage = in_conn_loss(t_sec, CONN_LOSS_WINDOWS) if flags["ENABLE_CONN_LOSS"] else False
                if outage:
                    recv_ok = False
                else:
                    recv_ok = (np.random.rand() > PACKET_LOSS_RATE) if flags["ENABLE_PACKET_LOSS"] else True

                kfL = kf_leader_dict[i]
                kfL.predict()
                if recv_ok:
                    kfL.update_optional(position=xL_true_del, velocity=vL_true_del, available=True)
                else:
                    kfL.update_optional(available=False)
                xL_hat, vL_hat = kfL.get_state()

                # ---------- Predecessor channel ----------
                # Truth
                xP_true = states[i-1][0]
                vP_true = states[i-1][3]

                # (A) Local sensor measurement (with noise/bias if enabled)
                obs_xp = xP_true
                obs_vp = vP_true
                if flags["ENABLE_MEAS_NOISE"]:
                    obs_xp += np.random.normal(0, POS_NOISE_STD)
                    obs_vp += np.random.normal(0, VEL_NOISE_STD)
                    applied_noise_pred = True
                if flags["ENABLE_MEAS_BIAS"]:
                    obs_xp += POS_BIAS
                    obs_vp += VEL_BIAS
                    applied_bias_pred = True

                # (B) Perfect communication (no faults)
                xP_comm = xP_true
                vP_comm = vP_true

                # KF: predict + sequential update (sensor → comm)
                kfP = kf_pred_dict[i]
                kfP.predict()

                # First update with sensor covariance
                R_backup = kfP.R.copy()
                kfP.R = R_MEAS
                kfP.update_optional(position=obs_xp, velocity=obs_vp, available=True)

                # Second update with comm covariance (very small, high confidence)
                kfP.R = R_COMM
                kfP.update_optional(position=xP_comm, velocity=vP_comm, available=True)

                # Restore default R for next iteration
                kfP.R = R_backup

                xP_hat, vP_hat = kfP.get_state()

                # Desired spacing (CTH policy)
                v_i = states[i][3]
                desired_spacing = D0 + H_HEADWAY * v_i

                # MPC with KF-estimated predecessor & leader
                a, delta = mpc.solve(
                    current_state=states[i],
                    lead_state=(xP_hat, 0.0, 0.0, vP_hat),
                    leader_state=(xL_hat, 0.0, 0.0, vL_hat),
                    desired_spacing=desired_spacing,
                    last_u=last_u[i]
                )

            # Vehicle dynamics update
            next_state = model.step(states[i], [a, delta])
            new_states.append(next_state)
            trajectories[i].append(next_state)
            last_u[i] = a

            # Logging
            state_log.append({
                "time": t_sec,
                "vehicle_id": i,
                "x": next_state[0],
                "y": next_state[1],
                "theta": next_state[2],
                "v": next_state[3],
                "a": a,
                "delta": delta,

                # Leader comm fault indicators
                "delay_active": bool(delay_active) if i > 0 else False,
                "outage": bool(outage) if i > 0 else False,
                "recv_ok": bool(recv_ok) if i > 0 else True,

                # Leader true / delayed / KF
                "xL_true": xL_true_now if i > 0 else np.nan,
                "vL_true": vL_true_now if i > 0 else np.nan,
                "xL_true_del": xL_true_del if i > 0 else np.nan,
                "vL_true_del": vL_true_del if i > 0 else np.nan,
                "xL_hat": xL_hat if i > 0 else np.nan,
                "vL_hat": vL_hat if i > 0 else np.nan,

                # Predecessor true / sensor / comm / KF
                "xP_true": xP_true if i > 0 else np.nan,
                "vP_true": vP_true if i > 0 else np.nan,
                "obs_xp":  obs_xp  if i > 0 else np.nan,
                "obs_vp":  obs_vp  if i > 0 else np.nan,
                "xP_comm": xP_comm if i > 0 else np.nan,
                "vP_comm": vP_comm if i > 0 else np.nan,
                "xP_hat":  xP_hat  if i > 0 else np.nan,
                "vP_hat":  vP_hat  if i > 0 else np.nan,

                # Flags for plotting
                "applied_bias_pred":  bool(applied_bias_pred) if i > 0 else False,
                "applied_noise_pred": bool(applied_noise_pred) if i > 0 else False,
            })

        states = new_states

    # -----------------------------
    # Save results
    # -----------------------------
    df = pd.DataFrame(state_log)
    csv_name = f"platoon_s{scenario}_{flags['NAME']}_delay{DELAY_SEC}_loss{PACKET_LOSS_RATE}.csv"
    df.to_csv(csv_name, index=False)
    print(f"Saved CSV: {csv_name}")

    # -----------------------------
    # Quick top-down trajectory plot
    # -----------------------------
    plt.figure(figsize=(10, 6))
    for i in range(NUM_VEHICLES):
        traj = np.array(trajectories[i])
        plt.plot(traj[:, 0], traj[:, 1], label=f"Car {i}")
    plt.axhline(y=1.5, color="gray", linestyle="--")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"CACC (Scenario {scenario}) — Ablation: {flags['NAME']}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"trajectory_{flags['NAME']}.png", dpi=300)
    plt.show()
