import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday
import ric_module

st.set_page_config(layout="wide")
st.title("🛰️ RIC Deviation Analyzer Dashboard V6")

# Sidebar Navigation
section = st.sidebar.radio("Navigation", [
    "📌 Overview",
    "📈 RIC Analyzer",
    "🛰️ Satellite Tracker",
    "🌍 World COP",
    "🕒 World Clock",
    "🚀 Rocket Launches"
])

# Load TLE data
TLE_FILE_PATH = "C:/Users/HP/Scripts/my_satellites.txt"
tle_dict = ric_module.load_tle_dict(TLE_FILE_PATH)

if section == "📌 Overview":
    st.header("📌 Overview")
    st.write("Welcome to the V6 Dashboard. Use the sidebar to explore RIC analysis, satellite tracking, and space operations tools.")

elif section == "📈 RIC Analyzer":
    # Sidebar Inputs
    master_id = st.sidebar.text_input("Enter Master NORAD ID").strip()
    target_id = st.sidebar.text_input("Enter Target NORAD ID").strip()
    forecast_days = st.sidebar.slider("Forecast Duration (days)", 1, 7, 3)
    threshold_km = st.sidebar.slider("Deviation Alert Threshold (km)", 1, 10000, 500)

    if master_id not in tle_dict or target_id not in tle_dict:
        st.warning("Please enter valid NORAD IDs present in your TLE file.")
        st.stop()

    # Satellite Init
    master_name, m1, m2 = tle_dict[master_id]
    target_name, t1, t2 = tle_dict[target_id]
    master_sat = Satrec.twoline2rv(m1, m2)
    target_sat = Satrec.twoline2rv(t1, t2)

    # Compute RIC
    start_time = datetime.utcnow()
    duration_minutes = forecast_days * 24 * 60
    times, ric, r1, r2 = ric_module.compute_ric(master_sat, target_sat, start_time, duration_minutes)

    # RIC Tabs
    tabs = st.tabs([
        "📊 RIC Plot",
        "📐 2D RIC Panels",
        "🌐 3D Orbits",
        "📄 CSV Export",
        "⚠️ Alerts",
        "🧭 3D RIC Deviation",
        "🌒 RIC 4-Pane Dark"
    ])

    with tabs[0]:
        df = pd.DataFrame({
            "UTC Time": times,
            "Radial (km)": ric[:, 0],
            "In-Track (km)": ric[:, 1],
            "Cross-Track (km)": ric[:, 2]
        }).set_index("UTC Time")
        st.line_chart(df)

    with tabs[1]:
        ric_module.display_ric_4panel_tab(times, ric, key="ric4_panel1")

    with tabs[2]:
        fig3d = ric_module.plot_3d_orbits(r1, r2, name1=master_name, name2=target_name)
        st.plotly_chart(fig3d, use_container_width=True)

    with tabs[3]:
        csv = df.reset_index().to_csv(index=False)
        st.download_button("📥 Download RIC CSV", data=csv, file_name=f"RIC_{master_id}_vs_{target_id}.csv")

    with tabs[4]:
        dist = (ric ** 2).sum(axis=1) ** 0.5
        min_dist = dist.min()
        st.metric("Minimum Distance (km)", f"{min_dist:.2f}")
        if (dist > threshold_km).any():
            st.error("🚨 Deviation threshold exceeded!")
        else:
            st.success("✅ All deviations within threshold")

    with tabs[5]:
        st.info("3D RIC deviation view coming soon.")

    with tabs[6]:
        ric_module.display_ric_4panel_tab(times, ric, key="ric4_panel6")

elif section == "🛰️ Satellite Tracker":
    st.header("🛰️ Satellite Tracker")
    st.info("This module will visualize satellite positions and motion in ECI/ECEF frames.")

elif section == "🌍 World COP":
    st.header("🌍 World COP Viewer")
    st.info("Strategic and tactical common operating picture (COP) displays will appear here.")

elif section == "🕒 World Clock":
    st.header("🕒 World Clock Display")
    st.info("Real-time multi-city world clock interface under development.")

elif section == "🚀 Rocket Launches":
    st.header("🚀 Rocket Launch Schedule")
    st.info("Live and upcoming rocket launches will be tracked and visualized here.")
