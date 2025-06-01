from sgp4.api import Satrec, jday
import numpy as np
import pymap3d as pm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import streamlit as st
import io

def load_tle_dict(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    tle_dict = {}
    i = 0
    while i < len(lines) - 2:
        if lines[i].startswith("0 "):
            name = lines[i][2:].strip()
            line1 = lines[i + 1]
            line2 = lines[i + 2]
            i += 3
        else:
            name = "Unknown"
            line1 = lines[i]
            line2 = lines[i + 1]
            i += 2

        try:
            raw_id = line1.split()[1]
            norad_id = ''.join(filter(str.isdigit, raw_id))
            tle_dict[norad_id] = (name, line1, line2)
        except Exception as e:
            print(f"Skipping malformed TLE: {e}")
            continue

    return tle_dict

def eci_to_ric(r_ref, v_ref, r_other):
    R_hat = r_ref / np.linalg.norm(r_ref)
    C_hat = np.cross(r_ref, v_ref)
    C_hat /= np.linalg.norm(C_hat)
    I_hat = np.cross(C_hat, R_hat)
    rot_matrix = np.vstack((R_hat, I_hat, C_hat)).T
    delta_r = r_other - r_ref
    return rot_matrix.T @ delta_r

def compute_ric(master_sat, target_sat, start_time, duration_minutes):
    times, ric_vals, eci_r1, eci_r2 = [], [], [], []
    for i in range(duration_minutes):
        t = start_time + timedelta(minutes=i)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
        e1, r1, v1 = master_sat.sgp4(jd, fr)
        e2, r2, v2 = target_sat.sgp4(jd, fr)
        if e1 == 0 and e2 == 0:
            ric = eci_to_ric(np.array(r1), np.array(v1), np.array(r2))
            ric_vals.append(ric)
            times.append(t)
            eci_r1.append(r1)
            eci_r2.append(r2)
    return np.array(times), np.array(ric_vals), np.array(eci_r1), np.array(eci_r2)

def eci_to_ecef_batch(eci_coords, time_list):
    ecef_coords = []
    for ri, t_utc in zip(eci_coords, time_list):
        try:
            x, y, z = pm.eci2ecef(t_utc, ri[0], ri[1], ri[2])
            ecef_coords.append([x / 1000.0, y / 1000.0, z / 1000.0])
        except Exception as e:
            print(f"ECI->ECEF conversion failed: {e}")
            continue
    return np.array(ecef_coords)

def plot_3d_orbits(r1, r2, name1="Master", name2="Target"):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=r1[:, 0], y=r1[:, 1], z=r1[:, 2],
        mode='lines',
        name=name1,
        line=dict(width=2)
    ))
    fig.add_trace(go.Scatter3d(
        x=r2[:, 0], y=r2[:, 1], z=r2[:, 2],
        mode='lines',
        name=name2,
        line=dict(width=2)
    ))
    fig.update_layout(
        title="3D Orbits",
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
            aspectmode="data"
        ),
        margin=dict(t=40, b=0)
    )
    return fig

def display_ric_4panel_tab(times, ric, key=None):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    start_idx, end_idx = st.slider("Select time range (minutes)", 0, len(times)-1, (0, len(times)-1), key=key)
    t = times[start_idx:end_idx+1]
    r, i, c = ric[start_idx:end_idx+1].T
    dist = np.linalg.norm(ric[start_idx:end_idx+1], axis=1)

    fig, axs = plt.subplots(2, 2, figsize=(14, 8), facecolor='black')
    labels = ["Cross vs In-Track", "Distance vs Time", "Radial vs Cross", "Radial vs In-Track"]
    colors = ['cyan', 'lime', 'orange', 'magenta']

    axs[0, 0].plot(i, c, color=colors[0], label=labels[0])
    axs[1, 0].plot(c, r, color=colors[2], label=labels[2])
    axs[1, 1].plot(i, r, color=colors[3], label=labels[3])
    axs[0, 1].plot(t, dist, color=colors[1], label=labels[1])
    axs[0, 1].scatter(t[np.argmin(dist)], np.min(dist), color='red', zorder=5)
    axs[0, 1].annotate(f"Min: {np.min(dist):.2f} km", xy=(t[np.argmin(dist)], np.min(dist)),
                      xytext=(15, 15), textcoords='offset points',
                      arrowprops=dict(arrowstyle='->', color='white'), fontsize=8, color='white')

    for ax, label, color in zip(axs.flat, labels, colors):
        ax.set_title(label, color='white')
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.set_facecolor('black')
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white')

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### 📋 RIC Deviation Summary")
    st.write({
        "Start Time": t[0].strftime('%Y-%m-%d %H:%M:%S'),
        "End Time": t[-1].strftime('%Y-%m-%d %H:%M:%S'),
        "Duration (min)": end_idx - start_idx + 1,
        "Min Distance (km)": float(np.min(dist)),
        "Max Distance (km)": float(np.max(dist)),
        "Avg Speed (km/min)": float(np.mean(np.linalg.norm(np.diff(ric[start_idx:end_idx+1], axis=0), axis=1)))
    })
