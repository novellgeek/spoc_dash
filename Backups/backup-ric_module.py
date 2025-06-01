
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import numpy as np
import pymap3d as pm
import matplotlib.pyplot as plt
import io
import streamlit as st
import plotly.express as px
import pandas as pd


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
        except:
            continue
    return tle_dict


def compute_ric(master_sat, target_sat, start_time, duration_minutes):
    times, ric_vals, eci_r1, eci_r2 = [], [], [], []
    for i in range(duration_minutes):
        t = start_time + timedelta(minutes=i)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
        e1, r1, v1 = master_sat.sgp4(jd, fr)
        e2, r2, v2 = target_sat.sgp4(jd, fr)
        if e1 == 0 and e2 == 0:
            r_ref = np.array(r1)
            v_ref = np.array(v1)
            r_other = np.array(r2)

            R_hat = r_ref / np.linalg.norm(r_ref)
            C_hat = np.cross(r_ref, v_ref)
            C_hat /= np.linalg.norm(C_hat)
            I_hat = np.cross(C_hat, R_hat)
            rot_matrix = np.vstack((R_hat, I_hat, C_hat)).T
            delta_r = r_other - r_ref
            ric = rot_matrix.T @ delta_r

            ric_vals.append(ric)
            times.append(t)
            eci_r1.append(r1)
            eci_r2.append(r2)
    return np.array(times), np.array(ric_vals), np.array(eci_r1), np.array(eci_r2)


def plot_3d_orbits(r1, r2, name1="Master", name2="Target"):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=r1[:, 0], y=r1[:, 1], z=r1[:, 2],
                               mode='lines', name=name1, line=dict(width=2)))
    fig.add_trace(go.Scatter3d(x=r2[:, 0], y=r2[:, 1], z=r2[:, 2],
                               mode='lines', name=name2, line=dict(width=2)))
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


def display_ric_4panel_tab(times, ric, key="ric4_panel", slider_enabled=False, sat_names=("Master", "Target")):
    st.subheader(f"📌 RIC Deviation: {sat_names[0]} vs {sat_names[1]}")

    if slider_enabled:
        start_idx, end_idx = st.slider(
            "Select time range (minutes)",
            0, len(times) - 1, (0, len(times) - 1),
            key=key
        )
        times = times[start_idx:end_idx + 1]
        ric = ric[start_idx:end_idx + 1]

    fig, axs = plt.subplots(2, 2, figsize=(14, 8), facecolor='black')
    fig.suptitle(f'RIC Deviation Components: {sat_names[0]} vs {sat_names[1]} (Dark Mode)', color='white')

    labels = ['In-Track vs Cross', 'Distance vs Time', 'Cross vs Radial', 'In-Track vs Radial']
    colors = ['cyan', 'orange', 'yellow', 'magenta']
    data = [
        (ric[:, 1], ric[:, 2]),
        (times, (ric ** 2).sum(axis=1) ** 0.5),
        (ric[:, 2], ric[:, 0]),
        (ric[:, 1], ric[:, 0])
    ]

    min_idx = np.argmin((ric ** 2).sum(axis=1) ** 0.5)
    min_time = times[min_idx]
    min_r, min_i, min_c = ric[min_idx]

    for ax, label, color, (x, y) in zip(axs.flat, labels, colors, data):
        ax.plot(x, y, '.', color=color)
        ax.set_title(label, color='white')
        ax.set_xlabel(label.split(' vs ')[0], color='white')
        ax.set_ylabel(label.split(' vs ')[1], color='white')
        ax.tick_params(axis='x', colors='gray')
        ax.tick_params(axis='y', colors='gray')
        ax.set_facecolor('#111111')
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

    axs[1, 0].annotate(f"Min Dist\n{min_time.strftime('%H:%M')}\n{(min_r**2+min_i**2+min_c**2)**0.5:.2f} km",
                      xy=(min_time, (min_r**2+min_i**2+min_c**2)**0.5),
                      xytext=(20, 20),
                      textcoords='offset points',
                      arrowprops=dict(facecolor='white', shrink=0.05),
                      fontsize=8, color='white')

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)

    dist = (ric ** 2).sum(axis=1) ** 0.5
    metrics = {
        'min': dist.min(),
        'max': dist.max(),
        'mean': dist.mean(),
        'png': buf.getvalue()
    }

    st.image(metrics['png'], caption=f"RIC 4-Panel Visualization for {sat_names[0]} vs {sat_names[1]}", use_container_width=True)
    st.markdown(f"**Min Distance:** {metrics['min']:.2f} km")
    st.markdown(f"**Max Distance:** {metrics['max']:.2f} km")
    st.markdown(f"**Mean Distance:** {metrics['mean']:.2f} km")

    st.download_button(
        label="📥 Download 4-Panel Plot (PNG)",
        data=metrics['png'],
        file_name=f"RIC_4Panel_{sat_names[0]}_vs_{sat_names[1]}.png",
        mime="image/png"
    )

    df = pd.DataFrame({
        "Time": times,
        "Radial (km)": ric[:, 0],
        "In-Track (km)": ric[:, 1],
        "Cross-Track (km)": ric[:, 2],
        "Distance (km)": dist
    })
    st.markdown("### 📊 Interactive Time Series of RIC Components")
    fig2 = px.line(df, x="Time", y=["Radial (km)", "In-Track (km)", "Cross-Track (km)", "Distance (km)"],
                   labels={"value": "Deviation (km)", "Time": "UTC Time"},
                   title=f"RIC Deviations Over Time: {sat_names[0]} vs {sat_names[1]}")
    st.plotly_chart(fig2, use_container_width=True)

    return fig, metrics
