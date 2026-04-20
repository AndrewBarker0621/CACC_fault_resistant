# vis_faults.py — fault-only visualization:
# fault timelines (per car / across cars) + leader true/delayed/observed/estimated comparisons.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ========= file =========
CSV_PATH = "platoon_s2_none_delay0.3_loss0.2.csv"
# CSV_PATH = "platoon_s3_conn_loss_delay0.3_loss0.2.csv"
save_dpi = 300
fault_vid_for_detail = 1      # specific follower

# color
COLOR = {
    0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red",
    4: "tab:purple", 5: "tab:brown", 6: "tab:pink", 7: "tab:gray"
}
def c(v): return COLOR.get(v, "gray")

def ensure_dir(p): os.makedirs(p, exist_ok=True)
ensure_dir("figs_faults")

# data
df = pd.read_csv(CSV_PATH)
vehicle_ids = sorted(df["vehicle_id"].unique())

# fault 
FAULT_KEYS = ["delay_active", "outage", "recv_ok", "meas_available", "applied_bias", "applied_noise"]
fault_cols = [k for k in FAULT_KEYS if k in df.columns]

# ====== A) single car ======
fdf = df[df["vehicle_id"] == fault_vid_for_detail].copy()
if (not fdf.empty) and fault_cols:
    plt.figure(figsize=(12,5))
    for k in fault_cols:
        y = fdf[k].astype(int).to_numpy()
        plt.step(fdf["time"], y, where="post", label=k)
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Time (s)"); plt.ylabel("On/Off")
    # plt.title(f"Fault flags (Car {fault_vid_for_detail})")
    plt.grid(True); plt.legend(ncol=3); plt.tight_layout()
    plt.savefig(f"figs_faults/fault_flags_car{fault_vid_for_detail}.png", dpi=save_dpi)

# ====== B) fault timeline ======
for k in fault_cols:
    plt.figure(figsize=(12,5))
    for vid in vehicle_ids:
        sdf = df[df["vehicle_id"] == vid]
        if k in sdf.columns:
            y = sdf[k].astype(int).to_numpy()
            t = sdf["time"].to_numpy()
            plt.step(t, y + 0.1*vid, where="post", label=f"Car {vid}")
    plt.xlabel("Time (s)"); plt.ylabel(f"{k} (offset by car id)")
    # plt.title(f"{k} timeline across vehicles")
    plt.grid(True); plt.legend(ncol=3); plt.tight_layout()
    plt.savefig(f"figs_faults/{k}_timeline_all_cars.png", dpi=save_dpi)

# ====== C) observation/estimation/ground truth (pos vs vel) ======
need_x = {"xL_true","xL_true_del","obs_x","xL_hat"}.issubset(df.columns)
need_v = {"vL_true","vL_true_del","obs_v","vL_hat"}.issubset(df.columns)

if not fdf.empty and need_x:
    plt.figure(figsize=(12,5))
    plt.plot(fdf["time"], fdf["xL_true"],     label="Leader True x", linewidth=2)
    plt.plot(fdf["time"], fdf["xL_true_del"], label="Leader True x (delayed)", linestyle="--")
    plt.plot(fdf["time"], fdf["xL_hat"],      label="KF Estimate x", linestyle="-.")
    obs_mask = ~np.isnan(fdf["obs_x"].to_numpy())
    plt.scatter(fdf.loc[obs_mask,"time"], fdf.loc[obs_mask,"obs_x"], s=10, label="Observed x")
    plt.xlabel("Time (s)"); plt.ylabel("Position (m)")
    # plt.title(f"Leader position: true / delayed / observed / KF (Car {fault_vid_for_detail})")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(f"figs_faults/leader_x_compare_car{fault_vid_for_detail}.png", dpi=save_dpi)

if not fdf.empty and need_v:
    plt.figure(figsize=(12,5))
    plt.plot(fdf["time"], fdf["vL_true"],     label="Leader True v", linewidth=2)
    plt.plot(fdf["time"], fdf["vL_true_del"], label="Leader True v (delayed)", linestyle="--")
    plt.plot(fdf["time"], fdf["vL_hat"],      label="KF Estimate v", linestyle="-.")
    obs_mask = ~np.isnan(fdf["obs_v"].to_numpy())
    plt.scatter(fdf.loc[obs_mask,"time"], fdf.loc[obs_mask,"obs_v"], s=10, label="Observed v")
    plt.xlabel("Time (s)"); plt.ylabel("Velocity (m/s)")
    # plt.title(f"Leader velocity: true / delayed / observed / KF (Car {fault_vid_for_detail})")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(f"figs_faults/leader_v_compare_car{fault_vid_for_detail}.png", dpi=save_dpi)

# ====== D) KF error ======
plt.figure(figsize=(12, 5))

# Position estimation error
pos_error = np.abs(fdf["xL_hat"] - fdf["xL_true"])
plt.plot(fdf["time"], pos_error, label="Position Estimation Error")

# Velocity estimation error
vel_error = np.abs(fdf["vL_hat"] - fdf["vL_true"])
plt.plot(fdf["time"], vel_error, label="Velocity Estimation Error")

plt.xlabel("Time (s)")
plt.ylabel("Error")
# plt.title("KF Estimation Error: Position and Velocity (Car 1)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("figs_faults/estimation_error_xv_car1.png", dpi=save_dpi)



print("Saved to ./figs_faults")
