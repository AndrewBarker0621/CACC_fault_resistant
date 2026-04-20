# idm_from_profile.py
# Simulate an IDM platoon driven by a given leader velocity profile.
# Outputs: figures + a CSV (compatible with vis_metrics.py style).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from idm import idm_acceleration  # uses your existing IDM model  :contentReference[oaicite:3]{index=3}

# -----------------------
# Config
# -----------------------
DT = 0.1                   # sampling time [s]
CAR_LENGTH = 5.0           # bumper-to-bumper gap = x_lead - x_ego - CAR_LENGTH
D0 = 5.0                   # standstill gap for CTG error
HEADWAY_T = 1.7            # time gap for CTG error
SAVE_DIR = "idm_figs"
CSV_OUT = os.path.join(SAVE_DIR, "idm_platoon_from_profile.csv")
os.makedirs(SAVE_DIR, exist_ok=True)

# Number of vehicles (including leader=0). You can change this.
NUM_VEHICLES = 5

# -----------------------
# 1) Load leader profile
# -----------------------
def load_leader_velocity_from_csv(path, time_col="time", v_col="v0"):
    """
    CSV format example:
      time, v0
      0.0,  15.2
      0.1,  15.3
      ...
    """
    df = pd.read_csv(path)
    t = df[time_col].to_numpy(dtype=float)
    v = df[v_col].to_numpy(dtype=float)
    return t, v

# If you want to pass arrays directly (e.g., from main_check in memory), use this function:
def set_leader_profile(time_array, v_array):
    t = np.asarray(time_array, dtype=float)
    v = np.asarray(v_array, dtype=float)
    assert t.ndim == 1 and v.ndim == 1 and len(t) == len(v)
    return t, v

# -----------------------
# 2) Simulate platoon with IDM followers
# -----------------------
def simulate_idm_platoon(time, v_lead, num_vehicles=NUM_VEHICLES,
                         init_gaps=None, init_speeds=None):
    """
    time: array [N], v_lead: array [N] leader speed
    init_gaps: list of initial bumper gaps (leader-1 gap, 1-2 gap, ...), length = num_vehicles-1
    init_speeds: list of initial speeds for each vehicle, length = num_vehicles
    """
    N = len(time)
    dt = np.median(np.diff(time))
    assert abs(dt - DT) < 1e-6, f"Expected DT={DT}, but got {dt}"

    # Leader position (integrate v), leader accel (numerical gradient)
    x_lead = np.zeros(N, dtype=float)
    for k in range(1, N):
        x_lead[k] = x_lead[k-1] + v_lead[k-1] * dt
    a_lead = np.gradient(v_lead, dt)

    # initial states
    if init_speeds is None:
        init_speeds = [v_lead[0]] + [max(v_lead[0] - 2.0, 0.0)] * (num_vehicles - 1)
    if init_gaps is None:
        # default bumper gaps between consecutive vehicles
        init_gaps = [20.0, 15.0, 12.0, 10.0][:max(0, num_vehicles-1)]

    # positions: place followers behind leader with given initial gaps
    x = np.zeros((num_vehicles, N), dtype=float)
    v = np.zeros((num_vehicles, N), dtype=float)
    a = np.zeros((num_vehicles, N), dtype=float)

    x[0, :] = x_lead
    v[0, :] = v_lead
    a[0, :] = a_lead

    # set follower initial pos/speed
    for i in range(1, num_vehicles):
        v[i, 0] = init_speeds[i]
        # head-to-head position: x[i-1,0] - init_gap - CAR_LENGTH
        x[i, 0] = x[ i-1, 0 ] - init_gaps[i-1] - CAR_LENGTH

    # rollout
    for k in range(1, N):
        # leader already set from profile
        for i in range(1, num_vehicles):
            # IDM acc uses bumper gap: d_i = x_lead - x_i - L
            a_i = idm_acceleration(
                v_i     = v[i, k-1],
                v_lead  = v[i-1, k-1],
                x_i     = x[i, k-1],
                x_lead  = x[i-1, k-1],
                L       = CAR_LENGTH
            )
            v[i, k] = max(v[i, k-1] + a_i * dt, 0.0)
            x[i, k] = x[i, k-1] + v[i, k] * dt
            a[i, k] = a_i

    return x, v, a

# -----------------------
# 3) Metrics (same definitions as vis_metrics.py)
# -----------------------
def moving_avg(x, k):
    if k <= 1:
        return x.copy()
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")

def compute_metrics(time, x, v, a,
                    car_length=CAR_LENGTH,
                    d0=D0, headway=HEADWAY_T,
                    ttc_cap=30.0, detrend_win_sec=5.0):
    """
    Returns dicts: gap_map, e_map, ttc_map, vd_map, ad_map
    Keys are vehicle indices (1..num_vehicles-1)
    """
    Nveh, N = x.shape
    dt = np.median(np.diff(time))
    win_len = max(1, int(round(detrend_win_sec/dt)))

    gap_map, e_map, ttc_map = {}, {}, {}
    vd_map, ad_map = {}, {}

    for vid in range(Nveh):
        # disturbances for all vehicles
        v_d = v[vid, :] - moving_avg(v[vid, :], win_len)
        a_d = a[vid, :] - moving_avg(a[vid, :], win_len)
        vd_map[vid] = (time, v_d)
        ad_map[vid] = (time, a_d)

    for vid in range(1, Nveh):
        # align with predecessor vid-1
        gap = x[vid-1, :] - x[vid, :] - car_length
        e   = gap - (d0 + headway * v[vid, :])

        dv  = v[vid, :] - v[vid-1, :]
        ttc = np.where(dv > 0.0, gap/np.maximum(dv, 1e-6), np.inf)
        ttc = np.minimum(ttc, ttc_cap)

        gap_map[vid] = (time, gap)
        e_map[vid]   = (time, e)
        ttc_map[vid] = (time, ttc)

    return gap_map, e_map, ttc_map, vd_map, ad_map

# -----------------------
# 4) Plot helpers
# -----------------------
def plot_series_map(series_map, ylabel, fname, ylim=None):
    plt.figure(figsize=(12,5))
    for vid, (t, y) in series_map.items():
        label = f"Vehicle {vid}" if vid != 0 else "Leader"
        plt.plot(t, y, label=label)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel("Time (s)"); plt.ylabel(ylabel)
    plt.grid(True); plt.legend(ncol=3); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, fname), dpi=300)
    plt.close()

# -----------------------
# 5) Main entry
# -----------------------
if __name__ == "__main__":
    # --- Option A: load leader v(t) from CSV exported by main_check ---
    # Example CSV with columns: time,v0
    # time, v0
    # 0.0,  15.3
    # 0.1,  15.4
    # ...
    # t, v_lead = load_leader_velocity_from_csv("leader_profile.csv", time_col="time", v_col="v0")

    # --- Option B: if you already have arrays from main_check in memory, use set_leader_profile ---
    # (Here we build a dummy placeholder; REPLACE with your real profile.)
    T_SIM = 100.0
    t = np.arange(0.0, T_SIM + 1e-9, DT)
    v0 = 15.0 + 2.0*np.sin(2*np.pi*0.02*t)  # <-- Replace with your main_check leader v(t)

    # Simulate
    x, v, a = simulate_idm_platoon(t, v0, num_vehicles=NUM_VEHICLES)

    # Build series maps (leader + followers)
    v_map = {0: (t, v[0,:])}
    a_map = {0: (t, a[0,:])}
    for vid in range(1, NUM_VEHICLES):
        v_map[vid] = (t, v[vid,:])
        a_map[vid] = (t, a[vid,:])

    gap_map, e_map, ttc_map, vd_map, ad_map = compute_metrics(
        t, x, v, a,
        car_length=CAR_LENGTH,
        d0=D0, headway=HEADWAY_T,
        ttc_cap=30.0, detrend_win_sec=5.0
    )

    # Save a CSV compatible with your vis_metrics workflow
    rows = []
    for vid in range(NUM_VEHICLES):
        for k, tk in enumerate(t):
            rows.append({"time": tk, "vehicle_id": vid,
                         "x": x[vid,k], "v": v[vid,k], "a": a[vid,k]})
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)

    # ---- Plots (same set you requested) ----
    # 1) Gap (followers only)
    plot_series_map(gap_map, "Gap (m)", "gap_over_time.png")

    # 2) Velocity (all)
    plot_series_map(v_map, "Velocity (m/s)", "velocity_over_time.png")

    # 3) Acceleration (all)
    plot_series_map(a_map, "Acceleration (m/s²)", "acceleration_over_time.png")

    # 4) Spacing error e (followers only)
    plot_series_map(e_map, "Spacing Error e (m)", "spacing_error_over_time.png")

    # 5) TTC (followers only)
    plot_series_map(ttc_map, "TTC (s)", "ttc_over_time.png", ylim=(0, 30))

    # 6) Velocity disturbances (all)
    plot_series_map(vd_map, "Velocity Disturbance ~v (m/s)", "velocity_disturbance_over_time.png")

    # 7) Acceleration disturbances (all)
    plot_series_map(ad_map, "Acceleration Disturbance ~a (m/s²)", "acceleration_disturbance_over_time.png")

    print(f"Saved figures to: {SAVE_DIR}")
    print(f"Saved CSV to: {CSV_OUT}")
