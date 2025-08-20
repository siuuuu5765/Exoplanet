import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import TimeSeries
from astropy import units as u
from astroquery.mast import Observations
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
import requests
import io
import warnings
import re
import random
from datetime import datetime
from lightkurve import search_lightcurve

warnings.filterwarnings("ignore")

# =============================
# Helper Functions
# =============================

@st.cache_data
def download_lightcurve(tic_id):
    try:
        search_result = search_lightcurve(f"TIC {tic_id}", mission="TESS")
        if len(search_result) == 0:
            return None
        lc = search_result.download()
        return lc
    except Exception as e:
        st.error(f"Failed to download lightcurve for TIC {tic_id}: {e}")
        return None

@st.cache_data
def get_random_exoplanet():
    try:
        results = NasaExoplanetArchive.query_criteria(
            table="pscomppars",
            select="pl_name, pl_orbper, pl_rade, pl_bmasse, pl_eqt, st_teff, st_rad, st_mass",
            pl_rade__gt=0.5, pl_rade__lt=4.0,
            st_teff__gt=3000, st_teff__lt=7000
        )
        if len(results) == 0:
            return None
        planet = results.to_pandas().sample(1).iloc[0]
        return planet
    except Exception as e:
        st.error(f"Failed to fetch exoplanet data; switching to random target. ({e})")
        return None

@st.cache_data
def get_research_sample():
    try:
        results = NasaExoplanetArchive.query_criteria(
            table="pscomppars",
            select="pl_name, pl_orbper, pl_rade, pl_bmasse, pl_eqt, st_teff, st_rad, st_mass, hostname",
            pl_rade__gt=0.5, pl_rade__lt=6.0,
            st_teff__gt=2500, st_teff__lt=8000
        )
        if len(results) == 0:
            return None
        return results.to_pandas()
    except Exception as e:
        st.error(f"Failed to fetch research data: {e}")
        return None

# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="TESS Planet Discovery Pro", layout="wide")
st.title("🔭 TESS Planet Discovery Pro")

st.sidebar.header("Target Selection")
target_option = st.sidebar.selectbox("Choose Target", ["Random Exoplanet", "Custom TIC ID", "Research Study Mode"])

if target_option == "Random Exoplanet":
    planet = get_random_exoplanet()
    if planet is not None:
        st.sidebar.success(f"Selected target: {planet['pl_name']} ({planet['pl_rade']} R⊕, {planet['pl_orbper']} d)")
        tic_id = planet.get("tic_id", None)
    else:
        tic_id = None

elif target_option == "Custom TIC ID":
    tic_id = st.sidebar.text_input("Enter TIC ID", value="307210830")

elif target_option == "Research Study Mode":
    sample = get_research_sample()
    if sample is not None:
        st.success("Loaded a research sample of confirmed TESS planets.")
        st.dataframe(sample.head(20))
        st.download_button("Download CSV", sample.to_csv(index=False), "tess_research_sample.csv")
    tic_id = None

if tic_id:
    lc = download_lightcurve(tic_id)
    if lc is not None:
        st.success(f"Lightcurve for TIC {tic_id} loaded successfully!")
        st.line_chart(lc.flux)
    else:
        st.error("Could not load lightcurve.")
else:
    if target_option != "Research Study Mode":
        st.warning("No valid target selected.")
