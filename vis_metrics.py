# vis_metrics.py — core KPIs only: trajectory/TTC/velocity/acceleration,
# spacing error e, velocity/acceleration disturbances, and L∞ gains.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ========= file =========
CSV_PATH = "platoon_s1_none_delay0.3_loss0.2.csv"

car_length = 5.0
headway_time = 1.7
standstill_gap = 5.0
save_dpi = 300

# window
detrend_win_sec = 5.0

# color and line type
COLOR = {
    0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red",
    4: "tab:purple", 5: "tab:brown", 6: "tab:pink", 7: "tab:gray"
}
LS = {0: "-", 1: "--", 2: "-.", 3: ":", 4: (0, (3,1,1,1))}
def c(v): return COLOR.get(v, "gray")
def ls(v): return LS.get(v, "-")

def ensure_dir(p): os.makedirs(p, exist_ok=True)
ensure_dir("figs_metrics")

# data
df = pd.read_csv(CSV_PATH)
vehicle_ids = sorted(df["vehicle_id"].unique())
t_unique = np.sort(df["time"].unique())
DT = np.median(np.diff(t_unique)) if len(t_unique) > 1 else 0.1
win_len = max(1, int(round(detrend_win_sec / DT)))

def moving_avg(x, k):
    if k <= 1: return x.copy()
    kernel = np.ones(k)/k
    return np.convolve(x, kernel, mode="same")

def align_ego_lead(ego_df, lead_df):
    ego = ego_df.copy().reset_index(drop=True)
    lead = lead_df.copy().reset_index(drop=True)
    n = min(len(ego), len(lead))
    return ego.iloc[:n].reset_index(drop=True), lead.iloc[:n].reset_index(drop=True)

# ------- v & a -------
v_map, a_map = {}, {}
for vid in vehicle_ids:
    veh = df[df["vehicle_id"] == vid].copy().reset_index(drop=True)
    t = veh["time"].to_numpy()
    v = veh["v"].to_numpy()
    if "a" in veh.columns:
        a = veh["a"].to_numpy()
    else:
        if len(t) >= 3:
            dt = np.gradient(t)
            a = np.gradient(v, dt)
        elif len(t) == 2:
            dt = max(t[1]-t[0], 1e-6)
            a = np.array([(v[1]-v[0])/dt, (v[1]-v[0])/dt])
        else:
            a = np.array([0.0])
    n = min(len(t), len(v), len(a))
    v_map[vid] = (t[:n], v[:n])
    a_map[vid] = (t[:n], a[:n])

# gap/e/TTC
gap_map, e_map, ttc_map = {}, {}, {}
ttc_max = 30.0
for vid in vehicle_ids:
    if vid == 0: continue
    ego = df[df["vehicle_id"] == vid]
    lead = df[df["vehicle_id"] == vid-1]
    ego, lead = align_ego_lead(ego, lead)

    t = ego["time"].to_numpy()
    x_ego, x_lead = ego["x"].to_numpy(), lead["x"].to_numpy()
    v_ego, v_lead = ego["v"].to_numpy(), lead["v"].to_numpy()

    gap = x_lead - x_ego - car_length
    gap_map[vid] = (t, gap)

    e = gap - (standstill_gap + headway_time * v_ego)
    e_map[vid] = (t, e)

    dv = v_ego - v_lead
    ttc = np.where(dv > 0, gap / np.maximum(dv, 1e-6), np.inf)
    ttc = np.minimum(ttc, ttc_max)
    ttc_map[vid] = (t, ttc)

# disturbance：\~v, \~a
vd_map, ad_map = {}, {}
for vid in vehicle_ids:
    t, v = v_map[vid]
    _, a = a_map[vid]
    vd = v - moving_avg(v, win_len)
    ad = a - moving_avg(a, win_len)
    vd_map[vid] = (t, vd)
    ad_map[vid] = (t, ad)

# ------- index -------
# 1) traj x vs time
plt.figure(figsize=(12,5))
for vid in vehicle_ids:
    veh = df[df["vehicle_id"] == vid]
    plt.plot(veh["x"], veh["time"], label=f"Car {vid}", color=c(vid), linestyle=ls(vid))
plt.xlabel("x (m)"); plt.ylabel("Time (s)")
plt.grid(True); plt.legend(ncol=3); plt.tight_layout()
plt.savefig("figs_metrics/trajectory_x_vs_time.png", dpi=save_dpi)

# 2) TTC
plt.figure(figsize=(12,5))
for vid,(t,ttc) in ttc_map.items():
    plt.plot(t, ttc, label=f"Car {vid}", color=c(vid), linestyle=ls(vid))
plt.xlabel("Time (s)"); plt.ylabel("TTC (s)")
plt.grid(True); plt.legend(ncol=3); plt.tight_layout()
plt.savefig("figs_metrics/ttc_over_time.png", dpi=save_dpi)

# 3) Velocity
plt.figure(figsize=(12,5))
for vid,(t,v) in v_map.items():
    plt.plot(t, v, label=f"Car {vid}", color=c(vid), linestyle=ls(vid))
plt.xlabel("Time (s)"); plt.ylabel("Velocity (m/s)")
plt.grid(True); plt.legend(ncol=3); plt.tight_layout()
plt.savefig("figs_metrics/velocity_over_time.png", dpi=save_dpi)

# 4) Acceleration
plt.figure(figsize=(12,5))
for vid,(t,a) in a_map.items():
    plt.plot(t, a, label=f"Car {vid}", color=c(vid), linestyle=ls(vid))
plt.xlabel("Time (s)"); plt.ylabel("Acceleration (m/s²)")
plt.grid(True); plt.legend(ncol=3); plt.tight_layout()
plt.savefig("figs_metrics/acceleration_over_time.png", dpi=save_dpi)

# 5) Spacing error e
plt.figure(figsize=(12,5))
for vid,(t,e) in e_map.items():
    plt.plot(t, e, label=f"Car {vid}", color=c(vid), linestyle=ls(vid))
plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
plt.xlabel("Time (s)"); plt.ylabel("Spacing Error e (m)")
plt.grid(True); plt.legend(ncol=3); plt.tight_layout()
plt.savefig("figs_metrics/spacing_error_over_time.png", dpi=save_dpi)

# 6) Velocity disturbance ~v
plt.figure(figsize=(12,5))
for vid,(t,vd) in vd_map.items():
    plt.plot(t, vd, label=f"Car {vid}", color=c(vid), linestyle=ls(vid))
plt.xlabel("Time (s)"); plt.ylabel("Velocity Disturbance ~v (m/s)")
plt.grid(True); plt.legend(ncol=3); plt.tight_layout()
plt.savefig("figs_metrics/velocity_disturbance_over_time.png", dpi=save_dpi)

# 7) Acceleration disturbance ~a
plt.figure(figsize=(12,5))
for vid,(t,ad) in ad_map.items():
    plt.plot(t, ad, label=f"Car {vid}", color=c(vid), linestyle=ls(vid))
plt.xlabel("Time (s)"); plt.ylabel("Acceleration Disturbance ~a (m/s²)")
plt.grid(True); plt.legend(ncol=3); plt.tight_layout()
plt.savefig("figs_metrics/acceleration_disturbance_over_time.png", dpi=save_dpi)

# ------- L∞ (max) -------
def max_abs(x):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    return np.max(np.abs(x)) if x.size else np.nan

def Linf_gain(series_map):
    rows = []
    for vid in vehicle_ids:
        if vid == 0: continue
        if (vid not in series_map) or ((vid-1) not in series_map): continue
        _, yi = series_map[vid]
        _, yp = series_map[vid-1]
        Mi, Mp = max_abs(yi), max_abs(yp)
        gain = np.nan if (Mp is np.nan or Mp == 0 or np.isnan(Mp)) else Mi / Mp
        rows.append({"vehicle_id": vid, "gain": gain})
    return pd.DataFrame(rows)

g_e  = Linf_gain(e_map)
g_vd = Linf_gain(vd_map)
g_ad = Linf_gain(ad_map)

# -- Linf table --
# follower id
followers = sorted([vid for vid in vehicle_ids if vid != 0])
table = pd.DataFrame({"vehicle_id": followers})

# vehicle_id projection
map_e  = dict(zip(g_e["vehicle_id"],  g_e["gain"]))
map_vd = dict(zip(g_vd["vehicle_id"], g_vd["gain"]))
map_ad = dict(zip(g_ad["vehicle_id"], g_ad["gain"]))

table["G_inf_e"]  = table["vehicle_id"].map(map_e)
table["G_inf_vd"] = table["vehicle_id"].map(map_vd)
table["G_inf_ad"] = table["vehicle_id"].map(map_ad)

# CSV
table.to_csv("figs_metrics/Linf_gains.csv", index=False)
print(table)

