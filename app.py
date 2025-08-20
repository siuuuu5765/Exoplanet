import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.timeseries import BoxLeastSquares
from astroquery.mast import Catalogs
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
import warnings
import re
from datetime import datetime
from lightkurve import search_lightcurve
import transitleastsquares as tls

# Optional heavy libs (only if available)
try:
    from sklearn.model_selection import train_test_split  # noqa: F401
    from sklearn.metrics import confusion_matrix, classification_report  # noqa: F401
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras.models import Sequential  # noqa: F401
    from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout  # noqa: F401
except Exception:
    pass

warnings.filterwarnings('ignore')

# =====================================================
# Streamlit page config
# =====================================================
st.set_page_config(
    page_title="TESS Planet Discovery Pro",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# Styles
# =====================================================
st.markdown(
    """
<style>
    .main-header { font-size: 2.5rem; color: #1E88E5; text-align: center; margin-bottom: 1rem; }
    .section-header { font-size: 1.5rem; color: #0D47A1; margin-top: 1.5rem; margin-bottom: 0.5rem; }
    .exoplanet-card { background-color: #f9f9f9; border-radius: 10px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .info-box { background-color: #E3F2FD; border-left: 5px solid #2196F3; padding: 10px; margin: 10px 0; }
    .chat-container { background-color: #f5f5f5; border-radius: 10px; padding: 15px; height: 400px; overflow-y: auto; margin-bottom: 10px; }
    .chat-message { padding: 8px 12px; border-radius: 18px; margin-bottom: 10px; max-width: 80%; }
    .user-message { background-color: #dcf8c6; align-self: flex-end; margin-left: auto; }
    .bot-message { background-color: #ffffff; align-self: flex-start; margin-right: auto; }
    .metric-container { display: flex; flex-wrap: wrap; gap: 15px; margin: 15px 0; }
    .metric-box { background-color: #f5f5f5; border-radius: 8px; padding: 15px; flex: 1; min-width: 200px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .metric-title { font-size: 0.9rem; color: #666; margin-bottom: 5px; }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #1E88E5; }
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# Session state init
# =====================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "sender": "bot",
            "message": "Hello! I'm ExoBot, your exoplanet discovery assistant. Ask me anything about TESS, exoplanets, or how to use this app!",
            "time": datetime.now().strftime("%H:%M"),
        }
    ]

if "research_results" not in st.session_state:
    st.session_state.research_results = None

if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = {}

if "candidate_rows" not in st.session_state:
    st.session_state.candidate_rows = []

# =====================================================
# Title / Intro
# =====================================================
st.markdown('<h1 class="main-header">🪐 TESS Planet Discovery Pro</h1>', unsafe_allow_html=True)
st.markdown(
    """
This advanced app helps you discover potential exoplanets using real data from NASA's **TESS**. 
It includes professional-grade algorithms, validation tools, and research components for student projects.
"""
)

# =====================================================
# Helper functions
# =====================================================
@st.cache_data(show_spinner=False)
def download_real_tess_data(tic_id: int):
    """Download, clean and flatten a TESS light curve for a TIC id.
    Returns time (days), flux (relative) or (None, None) on failure.
    """
    try:
        search = search_lightcurve(f"TIC {tic_id}", author="SPOC")
        if len(search) == 0:
            return None, None
        # download first matching LC; consider search.download_all().stitch() for longer baselines
        lc = search.download()
        lc = lc.remove_outliers(sigma=5).normalize()
        lc_flat = lc.flatten(window_length=101)
        return lc.time.value, lc_flat.flux.value
    except Exception:
        return None, None

@st.cache_data(show_spinner=False)
def get_star_metadata(tic_id: int):
    """Fetch stellar metadata. Try Exoplanet Archive (for known hosts) then TIC catalog via MAST.
    Returns dict with st_rad (R☉), st_teff (K), and any available fields.
    """
    # Defaults if not found
    meta = {"st_rad": None, "st_teff": None, "source": None}
    # 1) Exoplanet Archive (pscomppars is more consolidated)
    try:
        rows = NasaExoplanetArchive.query_criteria(
            table="pscomppars", tic_id=int(tic_id)
        )
        if len(rows) > 0:
            r0 = rows[0]
            meta["st_rad"] = (float(r0["st_rad"]) if not np.ma.is_masked(r0["st_rad"]) else None)
            meta["st_teff"] = (float(r0["st_teff"]) if not np.ma.is_masked(r0["st_teff"]) else None)
            meta["hostname"] = r0.get("hostname", None)
            meta["source"] = "ExoplanetArchive:pscomppars"
    except Exception:
        pass
    # 2) TIC catalog from MAST if needed
    if meta["st_rad"] is None or meta["st_teff"] is None:
        try:
            tic = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC")
            if len(tic) > 0:
                t0 = tic[0]
                # TIC uses 'rad' (R☉) and 'Teff'
                if meta["st_rad"] is None and "rad" in t0.colnames:
                    v = t0["rad"]
                    meta["st_rad"] = float(v) if v is not None and not np.ma.is_masked(v) else None
                if meta["st_teff"] is None and "Teff" in t0.colnames:
                    v = t0["Teff"]
                    meta["st_teff"] = float(v) if v is not None and not np.ma.is_masked(v) else None
                meta["Tmag"] = t0.get("Tmag", None)
                meta["ra"] = t0.get("ra", None)
                meta["dec"] = t0.get("dec", None)
                meta["source"] = meta.get("source") or "MAST:TIC"
        except Exception:
            pass
    return meta

# --- Detection algorithms ---

def bls_detection(time, flux):
    """Box Least Squares transit search returning best period, duration, t0, and periodogram."""
    model = BoxLeastSquares(time * u.day, flux)
    periods = np.logspace(-1, 1, 6000)  # 0.1–10 d, balanced grid
    durations = np.linspace(0.05, 0.2, 12)
    pg = model.autopower(periods, durations=durations)
    k = int(np.nanargmax(pg.power))
    return pg.period[k].value, pg.duration[k].value, pg.transit_time[k].value, pg


def tls_detection(time, flux):
    """Transit Least Squares search. Returns results object (cached upstream)."""
    model = tls.transitleastsquares(time, flux)
    results = model.power(
        period_min=0.5,
        period_max=20,
        transit_depth_min=0,
        oversampling_factor=3,
        duration_grid_step=1.05,
    )
    return results

# --- Validation ---

def calculate_signal_strength(time, flux, period, duration, t0):
    """Very simple binned SDE-like measure for bootstrap validation."""
    phases = ((time - t0) / period) % 1
    phase_bins = np.linspace(0, 1, 100)
    bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2

    binned = []
    for i in range(len(phase_bins) - 1):
        m = (phases >= phase_bins[i]) & (phases < phase_bins[i + 1])
        if np.any(m):
            binned.append(np.nanmean(flux[m]))
        else:
            binned.append(np.nan)
    binned = np.array(binned)

    in_tr = (bin_centers > (1 - duration / period)) | (bin_centers < duration / period)
    out_tr = ~in_tr
    if np.sum(in_tr) > 0 and np.sum(out_tr) > 0:
        transit_depth = np.nanmean(binned[out_tr]) - np.nanmean(binned[in_tr])
        denom = np.nanstd(binned[out_tr])
        return float(transit_depth / denom) if denom and not np.isnan(denom) else 0.0
    return 0.0


def validate_detection(time, flux, period, duration, t0, num_bootstrap=100):
    """Bootstrap on shuffled flux to estimate FAP and detection efficiency."""
    original = calculate_signal_strength(time, flux, period, duration, t0)
    boot = []
    rng = np.random.default_rng(42)
    for _ in range(num_bootstrap):
        shuffled = rng.permutation(flux)
        boot.append(calculate_signal_strength(time, shuffled, period, duration, t0))
    boot = np.array(boot)
    false_alarms = int(np.sum(boot >= original))
    fap = false_alarms / max(1, num_bootstrap)
    mean_boot = np.nanmean(boot)
    sde_eff = float(original / mean_boot) if mean_boot and not np.isnan(mean_boot) else 0.0
    return fap, sde_eff, boot.tolist()

# --- Multi-planet ---

def create_transit_model(time, period, duration, t0, depth=0.01):
    phases = ((time - t0) / period) % 1
    model = np.zeros_like(time, dtype=float)
    in_tr = (phases < duration / period) | (phases > (1 - duration / period))
    model[in_tr] = depth
    return model


def analyze_multi_planet_system(time, flux, base_depth=0.01):
    p1, d1, t01, pg1 = bls_detection(time, flux)
    depth1 = base_depth
    m1 = create_transit_model(time, p1, d1, t01, depth=depth1)
    res1 = flux - m1 + np.nanmedian(flux)

    p2, d2, t02, pg2 = bls_detection(time, res1)
    if np.nanmax(pg2.power) > 10:  # arbitrary threshold
        # try second removal
        depth2 = base_depth
        m2 = create_transit_model(time, p2, d2, t02, depth=depth2)
        res2 = res1 - m2 + np.nanmedian(res1)
        p3, d3, t03, pg3 = bls_detection(time, res2)
        if np.nanmax(pg3.power) > 10:
            return [
                {"period": p1, "duration": d1, "t0": t01, "depth": depth1},
                {"period": p2, "duration": d2, "t0": t02, "depth": depth2},
                {"period": p3, "duration": d3, "t0": t03, "depth": base_depth},
            ]
        return [
            {"period": p1, "duration": d1, "t0": t01, "depth": depth1},
            {"period": p2, "duration": d2, "t0": t02, "depth": depth2},
        ]
    return [{"period": p1, "duration": d1, "t0": t01, "depth": depth1}]

# --- Comparison with known planets ---

def compare_with_known_exoplanets(tic_id, detected_period, detected_radius):
    try:
        exo = NasaExoplanetArchive.query_criteria(table="ps", tic_id=int(tic_id))
        if len(exo) == 0:
            return None
        rows = []
        for row in exo:
            kp = row.get("pl_orbper", None)
            kr = row.get("pl_rade", None)
            name = row.get("pl_name", "Unknown")
            if kp and kr and not np.ma.is_masked(kp) and not np.ma.is_masked(kr):
                period_diff = abs(detected_period - float(kp)) / float(kp) * 100.0
                radius_diff = abs(detected_radius - float(kr)) / float(kr) * 100.0
                rows.append(
                    {
                        "name": name,
                        "known_period": float(kp),
                        "detected_period": float(detected_period),
                        "period_diff": period_diff,
                        "known_radius": float(kr),
                        "detected_radius": float(detected_radius),
                        "radius_diff": radius_diff,
                    }
                )
        return rows or None
    except Exception:
        return None

# --- Habitability ---

def assess_habitability(stellar_temp, stellar_radius, planet_period, planet_radius):
    # Kepler's third law with M* ~ R*^3 (very rough)
    stellar_mass = (stellar_radius or 1.0) ** 3
    a = (planet_period ** 2 * stellar_mass) ** (1 / 3)
    albedo = 0.3
    temp_eq = (stellar_temp or 5778) * ((1 - albedo) ** 0.25) * np.sqrt((stellar_radius or 1.0) / (2 * a))
    hz_inner, hz_outer = 0.95, 1.37
    in_hz = (a >= hz_inner) and (a <= hz_outer)
    is_rocky = planet_radius < 1.5
    esi = 0.0
    if in_hz and is_rocky:
        radius_similarity = 1 - abs(planet_radius - 1) / 1
        temp_similarity = 1 - abs(temp_eq - 288) / 288
        esi = max(0, min(1, radius_similarity * temp_similarity))
    return {
        "semi_major_axis": float(a),
        "equilibrium_temp": float(temp_eq),
        "in_habitable_zone": bool(in_hz),
        "is_rocky": bool(is_rocky),
        "esi": float(esi),
    }

# =====================================================
# Sidebar controls
# =====================================================
st.sidebar.title("🔍 Discovery Controls")
st.sidebar.markdown("### Select a Target Star")

selection = st.sidebar.selectbox(
    "Choose a target selection method:",
    ["Known Exoplanet Host", "Random TESS Target", "Custom TIC ID", "Research Study"],
)

tic_id = None
selected_host = None

if selection == "Known Exoplanet Host":
    try:
        exo = NasaExoplanetArchive.query_criteria(table="ps", default_flag=1, disc_telescope="TESS")
        hosts = np.unique(exo["hostname"]) if len(exo) else []
        selected_host = st.sidebar.selectbox("Select a host star:", hosts)
        host_rows = exo[exo["hostname"] == selected_host]
        # choose first with a TIC id, else None
        tic_candidates = [r.get("tic_id", None) for r in host_rows]
        tic_candidates = [int(x) for x in tic_candidates if x is not None and not np.ma.is_masked(x)]
        if len(tic_candidates):
            tic_id = tic_candidates[0]
            st.sidebar.success(f"Selected: {selected_host} (TIC {tic_id})")
            st.sidebar.markdown("### Known Planets")
            for r in host_rows:
                if not np.ma.is_masked(r.get("pl_orbper", None)):
                    st.sidebar.markdown(f"**{r.get('pl_name','?')}** — Period: {float(r['pl_orbper']):.2f} d, Radius: {float(r.get('pl_rade', np.nan)):.2f} R⊕")
        else:
            st.sidebar.warning("No TIC ID found for this host; try another.")
    except Exception as e:
        st.sidebar.error(f"Failed to fetch exoplanet data; switching to random target. ({e})")
        selection = "Random TESS Target"

elif selection == "Random TESS Target":
    tic_id = int(np.random.randint(100_000_000, 999_999_999))
    st.sidebar.success(f"Random TIC ID: {tic_id}")

elif selection == "Custom TIC ID":
    tic_id = int(
        st.sidebar.number_input(
            "Enter TESS Input Catalog (TIC) ID:", min_value=100_000_000, max_value=999_999_999, value=441420236, step=1
        )
    )

else:  # Research Study
    st.sidebar.info("Research study mode selected. Click 'Run Research Study' in the Research tab.")

# =====================================================
# Data acquisition (non-research)
# =====================================================
if selection != "Research Study" and tic_id:
    with st.spinner("Downloading real TESS data..."):
        time, flux = download_real_tess_data(tic_id)
        star_meta = get_star_metadata(tic_id)
else:
    time, flux, star_meta = None, None, {}

# =====================================================
# Main content
# =====================================================
if selection != "Research Study" and tic_id and time is not None:
    # Sidebar: star metadata
    st.sidebar.markdown("### ⭐ Star metadata")
    st.sidebar.write({k: v for k, v in star_meta.items() if k in ["hostname", "st_rad", "st_teff", "Tmag", "ra", "dec", "source"]})

    st.markdown('<h2 class="section-header">📊 Advanced Light Curve Analysis</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Light Curve",
        "Detection Methods",
        "Statistical Validation",
        "Multi-Planet Analysis",
        "Habitability Assessment",
        "Comparison with Known Planets",
    ])

    with tab1:
        st.markdown("### Raw Light Curve")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time, flux, "k.", markersize=1)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Relative Flux")
        ax.set_title(f"Real TESS Light Curve for TIC {tic_id}")
        ax.grid(True)
        st.pyplot(fig)
        st.markdown(
            '<div class="info-box">This is real TESS data processed to remove outliers and stellar variability. Dips in brightness may indicate planets passing in front of the star.</div>',
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown("### Advanced Detection Methods")
        # Cache TLS results to avoid recompute
        cache_key = f"tls_{tic_id}"
        if cache_key in st.session_state.analysis_cache:
            tls_res = st.session_state.analysis_cache[cache_key]
        else:
            tls_res = tls_detection(time, flux)
            st.session_state.analysis_cache[cache_key] = tls_res

        bls_period, bls_duration, bls_t0, bls_pg = bls_detection(time, flux)

        # TLS summary
        tls_period = float(tls_res.period)
        tls_depth = float(tls_res.depth) if hasattr(tls_res, "depth") and tls_res.depth is not None else np.nan
        tls_duration = float(tls_res.duration)
        tls_snr = float(getattr(tls_res, "SNR", getattr(tls_res, "sde", np.nan)))

        st.markdown("#### Box Least Squares (BLS) Results")
        st.write(f"Detected Period: {bls_period:.4f} days")
        st.write(f"Transit Duration: {bls_duration:.4f} days")
        st.write(f"Transit Midpoint (t0): {bls_t0:.4f} days")

        st.markdown("#### Transit Least Squares (TLS) Results")
        st.write(f"Detected Period: {tls_period:.4f} days")
        st.write(f"Transit Depth: {tls_depth:.6f}")
        st.write(f"Transit Duration: {tls_duration:.4f} days")
        st.write(f"Signal-to-Noise Ratio: {tls_snr:.2f}")

        # Periodograms
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        ax1.plot(bls_pg.period, bls_pg.power, "b-")
        ax1.set_xscale("log")
        ax1.set_xlabel("Period (days)")
        ax1.set_ylabel("BLS Power")
        ax1.set_title("BLS Periodogram")
        ax1.grid(True)

        ax2.plot(tls_res.period, tls_res.SDE, "r-")
        ax2.set_xscale("log")
        ax2.set_xlabel("Period (days)")
        ax2.set_ylabel("TLS SDE")
        ax2.set_title("TLS Periodogram")
        ax2.grid(True)

        plt.tight_layout()
        st.pyplot(fig)

        # Planet radius estimate using star radius if available
        stellar_radius = star_meta.get("st_rad", 1.0) or 1.0  # R☉ default
        stellar_radius_earth = stellar_radius * 109.0
        transit_depth = tls_depth if np.isfinite(tls_depth) and tls_depth > 0 else 0.01
        planet_radius_estimate = stellar_radius_earth * np.sqrt(transit_depth)

        st.markdown("### Estimated Planet Properties")
        colA, colB, colC = st.columns(3)
        colA.metric("Orbital Period", f"{tls_period:.4f} days")
        colB.metric("Planet Radius", f"{planet_radius_estimate:.2f} Earth radii")
        colC.metric("Signal-to-Noise Ratio", f"{tls_snr:.2f}")

        if planet_radius_estimate < 1.0:
            planet_class = "Sub-Earth"
        elif planet_radius_estimate < 1.25:
            planet_class = "Earth-sized"
        elif planet_radius_estimate < 2.0:
            planet_class = "Super-Earth"
        elif planet_radius_estimate < 4.0:
            planet_class = "Neptune-sized"
        else:
            planet_class = "Gas Giant"
        st.success(f"Planet Classification: {planet_class}")

        # Save candidate button
        with st.expander("Save this candidate"):
            notes = st.text_input("Notes (optional)", key="notes")
            if st.button("Add to Candidate List"):
                st.session_state.candidate_rows.append(
                    {
                        "tic_id": tic_id,
                        "period_days": tls_period,
                        "radius_re": planet_radius_estimate,
                        "snr": tls_snr,
                        "star_radius_rsun": stellar_radius,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "notes": notes,
                    }
                )
                st.success("Added. See 'Candidates' section at the bottom to download CSV.")

    with tab3:
        st.markdown("### Statistical Validation")
        fap, sde_eff, boot = validate_detection(time, flux, tls_period, tls_duration, bls_t0, num_bootstrap=100)
        st.markdown("#### Bootstrap Analysis Results")
        st.write(f"False Alarm Probability: {fap:.4f}")
        st.write(f"Signal Detection Efficiency: {sde_eff:.2f}")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(boot, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(x=sde_eff, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Signal Strength')
        ax.set_ylabel('Frequency')
        ax.set_title('Bootstrap Distribution of Signal Strengths')
        ax.legend(['Detected Signal', 'Bootstrap Distribution'])
        ax.grid(True)
        st.pyplot(fig)
        if fap < 0.01:
            st.success("✅ Detection is statistically significant (FAP < 1%)")
        elif fap < 0.05:
            st.warning("⚠️ Detection is marginally significant (1% ≤ FAP < 5%)")
        else:
            st.error("❌ Detection is not statistically significant (FAP ≥ 5%)")

    with tab4:
        st.markdown("### Multi-Planet System Analysis")
        planets = analyze_multi_planet_system(time, flux, base_depth=transit_depth)
        st.write(f"Detected {len(planets)} potential planet(s) in this system:")
        for i, p in enumerate(planets, start=1):
            st.markdown(f"#### Planet {i}")
            st.write(f"Orbital Period: {p['period']:.4f} days")
            st.write(f"Transit Duration: {p['duration']:.4f} days")
            st.write(f"Transit Midpoint: {p['t0']:.4f} days")
            # Estimate radius using detected depth (fallback to base)
            depth_i = p.get("depth", transit_depth)
            pr_i = stellar_radius_earth * np.sqrt(max(depth_i, 1e-4))
            st.write(f"Estimated Radius: {pr_i:.2f} Earth radii")
            if pr_i < 1.0:
                cls = "Sub-Earth"
            elif pr_i < 1.25:
                cls = "Earth-sized"
            elif pr_i < 2.0:
                cls = "Super-Earth"
            elif pr_i < 4.0:
                cls = "Neptune-sized"
            else:
                cls = "Gas Giant"
            st.write(f"Classification: {cls}")
            st.markdown("---")

    with tab5:
        st.markdown("### Habitability Assessment")
        stellar_temp = star_meta.get("st_teff", 5778) or 5778
        stellar_radius = star_meta.get("st_rad", 1.0) or 1.0
        habitability = assess_habitability(stellar_temp, stellar_radius, tls_period, planet_radius_estimate)
        st.markdown("#### Habitability Metrics")
        st.write(f"Semi-major Axis: {habitability['semi_major_axis']:.3f} AU")
        st.write(f"Equilibrium Temperature: {habitability['equilibrium_temp']:.1f} K")
        st.write(f"In Habitable Zone: {'Yes' if habitability['in_habitable_zone'] else 'No'}")
        st.write(f"Rocky Planet: {'Yes' if habitability['is_rocky'] else 'No'}")
        st.write(f"Earth Similarity Index: {habitability['esi']:.3f}")

        # Simple HZ sketch
        fig, ax = plt.subplots(figsize=(10, 6))
        star = plt.Circle((0, 0), 0.1, color='yellow', zorder=10)
        ax.add_patch(star)
        hz_inner = plt.Circle((0, 0), 0.95, color='none', edgecolor='green', linestyle='--', linewidth=2)
        hz_outer = plt.Circle((0, 0), 1.37, color='none', edgecolor='green', linestyle='--', linewidth=2)
        ax.add_patch(hz_inner)
        ax.add_patch(hz_outer)
        planet_circle = plt.Circle((habitability['semi_major_axis'], 0), 0.05, color='blue' if habitability['in_habitable_zone'] else 'red', zorder=5)
        ax.add_patch(planet_circle)
        ax.set_xlim(-0.2, 2.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('Distance from Star (AU)')
        ax.set_title('Habitability Assessment')
        ax.grid(True)
        ax.legend(['Star', 'Habitable Zone', 'Planet'], loc='upper right')
        st.pyplot(fig)

        if habitability['in_habitable_zone'] and habitability['is_rocky']:
            st.success("🌍 This planet is a potentially habitable rocky world!")
        elif habitability['in_habitable_zone']:
            st.warning("🌊 This planet is in the habitable zone but may not be rocky.")
        else:
            st.info("🔴 This planet is not in the habitable zone.")

    with tab6:
        st.markdown("### Comparison with Known Exoplanets")
        comparison_results = compare_with_known_exoplanets(tic_id, tls_period, planet_radius_estimate)
        if comparison_results:
            st.write("Comparison with known planets in this system:")
            for r in comparison_results:
                st.markdown(f"#### {r['name']}")
                st.write(f"Known Period: {r['known_period']:.4f} days")
                st.write(f"Detected Period: {r['detected_period']:.4f} days")
                st.write(f"Period Difference: {r['period_diff']:.2f}%")
                st.write(f"Known Radius: {r['known_radius']:.2f} Earth radii")
                st.write(f"Detected Radius: {r['detected_radius']:.2f} Earth radii")
                st.write(f"Radius Difference: {r['radius_diff']:.2f}%")
                if r['period_diff'] < 5 and r['radius_diff'] < 20:
                    st.success("✅ Excellent match with known planet!")
                elif r['period_diff'] < 10 and r['radius_diff'] < 40:
                    st.warning("⚠️ Good match with known planet.")
                else:
                    st.error("❌ Poor match with known planet.")
                st.markdown("---")
        else:
            st.info("No known exoplanets in this system for comparison. This could be a new discovery!")
            if st.button("Confirm Discovery", key="confirm_discovery"):
                st.balloons()
                st.success(
                    f"🎉 Congratulations! You've discovered a potential {planet_class} exoplanet orbiting TIC {tic_id}!"
                )
                st.markdown(
                    f"""
**Discovery Summary:**
- **Host Star:** TIC {tic_id}
- **Planet Period:** {tls_period:.4f} days
- **Planet Radius:** {planet_radius_estimate:.2f} Earth radii
- **Signal-to-Noise Ratio:** {tls_snr:.2f}
- **False Alarm Probability:** {fap:.4f}
- **Planet Class:** {planet_class}
- **Habitability:** {'Potentially habitable' if habitability['in_habitable_zone'] and habitability['is_rocky'] else 'Not habitable'}
                    """
                )

# =====================================================
# Research Study
# =====================================================
if selection == "Research Study":
    st.markdown('<h2 class="section-header">🔬 Research Study: Detection Method Comparison</h2>', unsafe_allow_html=True)
    st.markdown(
        """
This research study compares the performance of **Box Least Squares (BLS)** and **Transit Least Squares (TLS)** 
for detecting exoplanets in TESS data. We analyze a sample of known exoplanet systems and evaluate accuracy.
"""
    )

    @st.cache_data(show_spinner=False)
    def conduct_research_study():
        try:
            exoplanets = NasaExoplanetArchive.query_criteria(
                table="ps",
                default_flag=1,
                disc_telescope="TESS",
                pl_rade__gt=0.5,
                pl_rade__lt=4.0,
            )
            sample_size = int(min(30, len(exoplanets)))
            if sample_size == 0:
                return None, None
            rng = np.random.default_rng(123)
            idx = rng.choice(len(exoplanets), sample_size, replace=False)
            sample = exoplanets[idx]
            rows = []
            for planet in sample:
                tic_val = planet.get("tic_id", None)
                if tic_val is None or np.ma.is_masked(tic_val):
                    continue
                tic_local = int(tic_val)
                time, flux = download_real_tess_data(tic_local)
                if time is None:
                    continue
                bls_p, _, _, _ = bls_detection(time, flux)
                tls_res = tls_detection(time, flux)
                known_p = float(planet.get("pl_orbper", np.nan))
                known_r = float(planet.get("pl_rade", np.nan))
                tls_p = float(tls_res.period)
                tls_snr = float(getattr(tls_res, "SNR", getattr(tls_res, "sde", np.nan)))
                rows.append(
                    {
                        "tic_id": tic_local,
                        "known_period": known_p,
                        "known_radius": known_r,
                        "bls_period": bls_p,
                        "bls_period_error": abs(bls_p - known_p) / known_p * 100 if known_p else np.nan,
                        "tls_period": tls_p,
                        "tls_period_error": abs(tls_p - known_p) / known_p * 100 if known_p else np.nan,
                        "tls_snr": tls_snr,
                    }
                )
            if not rows:
                return None, None
            df = pd.DataFrame(rows)
            bls_mean_error = float(df["bls_period_error"].dropna().mean()) if df["bls_period_error"].notna().any() else np.nan
            tls_mean_error = float(df["tls_period_error"].dropna().mean()) if df["tls_period_error"].notna().any() else np.nan
            comparison = {
                "bls_mean_error": bls_mean_error,
                "tls_mean_error": tls_mean_error,
                "better_method": ("TLS" if (np.nan_to_num(tls_mean_error) < np.nan_to_num(bls_mean_error)) else "BLS"),
                "detection_rate": float((df["tls_snr"] > 7).mean() * 100.0),
            }
            return comparison, df
        except Exception:
            return None, None

    if st.button("Run Research Study", key="run_research"):
        with st.spinner("Conducting research study..."):
            st.session_state.research_results = conduct_research_study()

    if st.session_state.research_results:
        comparison, df = st.session_state.research_results
        if comparison is not None:
            st.markdown('<div class="exoplanet-card">', unsafe_allow_html=True)
            st.markdown("### Research Results Summary")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(
                f"""
<div class="metric-box"><div class="metric-title">BLS Mean Error</div><div class="metric-value">{comparison['bls_mean_error']:.2f}%</div></div>
<div class="metric-box"><div class="metric-title">TLS Mean Error</div><div class="metric-value">{comparison['tls_mean_error']:.2f}%</div></div>
<div class="metric-box"><div class="metric-title">Better Method</div><div class="metric-value">{comparison['better_method']}</div></div>
<div class="metric-box"><div class="metric-title">Detection Rate</div><div class="metric-value">{comparison['detection_rate']:.1f}%</div></div>
""",
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("### Detailed Results")
            st.dataframe(df)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            ax1.hist(df['bls_period_error'].dropna(), alpha=0.5, label='BLS', bins=20)
            ax1.hist(df['tls_period_error'].dropna(), alpha=0.5, label='TLS', bins=20)
            ax1.set_xlabel('Period Error (%)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Period Error Distribution')
            ax1.legend()
            ax1.grid(True)

            ax2.hist(df['tls_snr'].dropna(), bins=20)
            ax2.axvline(x=7, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('Signal-to-Noise Ratio')
            ax2.set_ylabel('Frequency')
            ax2.set_title('TLS SNR Distribution')
            ax2.legend(['Detection Threshold', 'SNR'])
            ax2.grid(True)

            plt.tight_layout()
            st.pyplot(fig)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Research Data",
                data=csv,
                file_name='tess_detection_comparison.csv',
                mime='text/csv'
            )
        else:
            st.error("Research study failed. Please try again.")

# =====================================================
# Chatbot
# =====================================================
st.markdown('<h2 class="section-header">🤖 ExoBot - Your Exoplanet Assistant</h2>', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        if chat["sender"] == "user":
            st.markdown(f"<div class='chat-message user-message'>{chat['message']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-message bot-message'><b>ExoBot:</b> {chat['message']}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

user_input = st.text_input("Ask ExoBot a question:", key="chat_input")


def get_bot_response(user_input):
    inp = (user_input or "").lower()
    responses = {
        r"hello|hi|hey": "Hello there! I'm ExoBot. How can I help you today?",
        r"tess|transiting exoplanet survey satellite": "TESS is a NASA space telescope (launched 2018) that searches for exoplanets via the transit method.",
        r"exoplanet|planet": "Exoplanets are planets orbiting other stars. Over 5,000 have been confirmed so far.",
        r"transit method": "Transit method measures tiny dips in a star's brightness when a planet crosses in front of it.",
        r"light curve": "A light curve plots brightness vs time. Dips can indicate transits.",
        r"periodogram": "A periodogram shows the power of periodic signals at different periods.",
        r"habitable zone|goldilocks": "The habitable zone allows liquid water on a planet's surface given certain assumptions.",
        r"how.*discover|find.*planet": "Use this app: choose a target, inspect dips, run BLS/TLS, fold & validate, then estimate properties.",
        r"false positive": "False positives can come from eclipsing binaries, spots, or systematics. Validation helps rule them out.",
        r"thanks|thank you": "You're welcome!"
    }
    for pat, resp in responses.items():
        if re.search(pat, inp):
            return resp
    return "Great question! Ask me about TESS, transits, detection algorithms, or how to use this app."

if st.button("Send", key="send_button") and user_input:
    st.session_state.chat_history.append({"sender": "user", "message": user_input, "time": datetime.now().strftime("%H:%M")})
    st.session_state.chat_history.append({"sender": "bot", "message": get_bot_response(user_input), "time": datetime.now().strftime("%H:%M")})
    st.session_state.chat_input = ""
    st.experimental_rerun()

# =====================================================
# Candidates persistence (CSV)
# =====================================================
st.markdown('<h2 class="section-header">🗂️ Candidates</h2>', unsafe_allow_html=True)
if st.session_state.candidate_rows:
    cand_df = pd.DataFrame(st.session_state.candidate_rows)
    st.dataframe(cand_df)
    st.download_button(
        label="Download Candidates CSV",
        data=cand_df.to_csv(index=False).encode('utf-8'),
        file_name=f"tess_candidates_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv",
        mime='text/csv',
    )
else:
    st.info("No saved candidates yet. Save one from the Detection tab.")

# =====================================================
# About
# =====================================================
st.markdown("---")
st.markdown("Made with ❤️ using TESS data and Streamlit")
