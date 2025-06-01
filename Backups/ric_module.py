from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import numpy as np
import pymap3d as pm
import matplotlib.pyplot as plt
import io
import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go


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
            raw_id = line1.split()[1]  # e.g., "00011U"
            norad_id = ''.join(filter(str.isdigit, raw_id))  # extract only digits
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
            ecef_coords.append([x / 1000.0, y / 1000.0, z / 1000.0])  # convert to km
        except Exception as e:
            print(f"ECI->ECEF conversion failed: {e}")
            continue
    return np.array(ecef_coords)


def plot_3d_orbits(r1, r2, name1="Master", name2="Target"):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=r1[:, 0], y=r1[:, 1], z=r1[:, 2],
        mode='lines', name=name1, line=dict(width=2)
    ))
    fig.add_trace(go.Scatter3d(
        x=r2[:, 0], y=r2[:, 1], z=r2[:, 2],
        mode='lines', name=name2, line=dict(width=2)
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


def display_ric_4panel_tab(times, ric, key="ric4_panel", slider_enabled=True, sat_names=("Master", "Target")):
    display_cross_intrack_panels(times, ric, key=key, sat_names=sat_names)


def display_cross_intrack_panels(times, ric, key="cross_intrack", sat_names=("Master", "Target")):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=120)
    fig.patch.set_facecolor("black")
    fig.suptitle(f"RIC Deviation Panels: {sat_names[0]} vs {sat_names[1]}", fontsize=12, color='white')
    axs = axs.flatten()

    dist = np.linalg.norm(ric, axis=1)
    min_idx = np.argmin(dist)

    axs[0].plot(times, ric[:, 0], color='cyan')
    axs[0].set_title("Radial Component", color='white', fontsize=10)
    axs[0].set_ylabel("km", color='white', fontsize=9)
    axs[0].annotate(f"Min: {ric[min_idx, 0]:.2f} km", xy=(times[min_idx], ric[min_idx, 0]),
                    xytext=(10, 10), textcoords='offset points', color='white', fontsize=8,
                    arrowprops=dict(color='white'))

    axs[1].plot(ric[:, 1], ric[:, 2], color='lime')
    axs[1].set_title("In-Track vs Cross-Track", color='white', fontsize=10)
    axs[1].set_xlabel("In-Track (km)", color='white', fontsize=9)
    axs[1].set_ylabel("Cross-Track (km)", color='white', fontsize=9)
    axs[1].annotate(f"Min Dist: {dist[min_idx]:.2f} km", xy=(ric[min_idx, 1], ric[min_idx, 2]),
                    xytext=(10, -20), textcoords='offset points', color='white', fontsize=8,
                    arrowprops=dict(color='white'))

    axs[2].plot(times, ric[:, 1], color='orange')
    axs[2].set_title("In-Track Component", color='white', fontsize=10)
    axs[2].set_ylabel("km", color='white', fontsize=9)
    axs[2].annotate(f"Min: {ric[min_idx, 1]:.2f} km", xy=(times[min_idx], ric[min_idx, 1]),
                    xytext=(10, 10), textcoords='offset points', color='white', fontsize=8,
                    arrowprops=dict(color='white'))

    axs[3].plot(times, ric[:, 2], color='magenta')
    axs[3].set_title("Cross-Track Component", color='white', fontsize=10)
    axs[3].set_ylabel("km", color='white', fontsize=9)
    axs[3].annotate(f"Min: {ric[min_idx, 2]:.2f} km", xy=(times[min_idx], ric[min_idx, 2]),
                    xytext=(10, 10), textcoords='offset points', color='white', fontsize=8,
                    arrowprops=dict(color='white'))

    for ax in axs:
        ax.tick_params(colors='white', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_facecolor("black")

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    st.image(buf, use_container_width=True)
    plt.close(fig)


def display_ric_plot_tab(times, ric, key="ric_plot", sat_names=("Master", "Target")):
    df = pd.DataFrame({
        "UTC Time": times,
        "Radial (km)": ric[:, 0],
        "In-Track (km)": ric[:, 1],
        "Cross-Track (km)": ric[:, 2]
    })
    fig = px.line(df, x="UTC Time", y=["Radial (km)", "In-Track (km)", "Cross-Track (km)"],
                  labels={"value": "Distance (km)"},
                  title=f"Interactive RIC Time Series: {sat_names[0]} vs {sat_names[1]}")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    dist = np.linalg.norm(ric, axis=1)
    min_dist = dist.min()
    min_idx = np.argmin(dist)
    st.markdown(f"""
    ### 📌 RIC Summary: {sat_names[0]} vs {sat_names[1]}

    - **Minimum Distance:** `{dist[min_idx]:.2f} km` at `{times[min_idx].strftime('%Y-%m-%d %H:%M:%S')} UTC`
    - **Peak Radial Deviation:** `{np.max(np.abs(ric[:,0])):.2f} km`
    - **Peak In-Track Deviation:** `{np.max(np.abs(ric[:,1])):.2f} km`
    - **Peak Cross-Track Deviation:** `{np.max(np.abs(ric[:,2])):.2f} km`
    - **Total Time Steps Analyzed:** `{len(times)}`
    """)
