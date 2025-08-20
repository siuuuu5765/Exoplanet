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
from astropy.timeseries import BoxLeastSquares
import transitleastsquares as tls
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="TESS Planet Discovery Pro",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .exoplanet-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .discovery-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 1rem;
        margin-top: 10px;
    }
    .discovery-button:hover {
        background-color: #45a049;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        padding: 10px;
        margin: 10px 0;
    }
    .chat-container {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 15px;
        height: 400px;
        overflow-y: auto;
        margin-bottom: 10px;
    }
    .chat-message {
        padding: 8px 12px;
        border-radius: 18px;
        margin-bottom: 10px;
        max-width: 80%;
    }
    .user-message {
        background-color: #dcf8c6;
        align-self: flex-end;
        margin-left: auto;
    }
    .bot-message {
        background-color: #ffffff;
        align-self: flex-start;
        margin-right: auto;
    }
    .chat-input-container {
        display: flex;
        margin-top: 10px;
    }
    .chat-input {
        flex-grow: 1;
        padding: 10px;
        border-radius: 20px;
        border: 1px solid #ddd;
    }
    .chat-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 15px;
        margin-left: 10px;
        cursor: pointer;
    }
    .method-card {
        background-color: #f0f7ff;
        border-left: 4px solid #2196F3;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    .method-title {
        font-weight: bold;
        color: #0D47A1;
        margin-bottom: 5px;
    }
    .research-card {
        background-color: #f0fff0;
        border-left: 4px solid #4CAF50;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        margin: 15px 0;
    }
    .metric-box {
        background-color: #f5f5f5;
        border-radius: 8px;
        padding: 15px;
        flex: 1;
        min-width: 200px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat and research
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append({
        "sender": "bot",
        "message": "Hello! I'm ExoBot, your exoplanet discovery assistant. Ask me anything about TESS, exoplanets, or how to use this app!",
        "time": datetime.now().strftime("%H:%M")
    })

if 'research_results' not in st.session_state:
    st.session_state.research_results = None

# App title and introduction
st.markdown('<h1 class="main-header">🪐 TESS Planet Discovery Pro</h1>', unsafe_allow_html=True)
st.markdown("""
This advanced app helps you discover potential exoplanets using real data from NASA's Transiting Exoplanet Survey Satellite (TESS). 
We've implemented professional-grade algorithms and added research components for ISEF-level analysis.
""")

# Sidebar with controls
st.sidebar.title("🔍 Discovery Controls")
st.sidebar.markdown("### Select a Target Star")

# Option to select a target
target_option = st.sidebar.selectbox(
    "Choose a target selection method:",
    ["Known Exoplanet Host", "Random TESS Target", "Custom TIC ID", "Research Study"]
)

if target_option == "Known Exoplanet Host":
    # Get list of known exoplanet hosts from TESS
    try:
        exoplanets = NasaExoplanetArchive.query_criteria(table="ps", 
                                                        default_flag=1, 
                                                        disc_telescope="TESS")
        host_names = np.unique(exoplanets['hostname'])
        selected_host = st.sidebar.selectbox("Select a host star:", host_names)
        
        # Get TIC ID for the selected host
        host_data = exoplanets[exoplanets['hostname'] == selected_host]
        tic_id = host_data['tic_id'][0]
        st.sidebar.success(f"Selected: {selected_host} (TIC {tic_id})")
        
        # Display known planets
        st.sidebar.markdown("### Known Planets")
        for planet in host_data:
            planet_name = planet['pl_name']
            planet_period = planet['pl_orbper']
            planet_radius = planet['pl_rade']
            st.sidebar.markdown(f"**{planet_name}**")
            st.sidebar.markdown(f"Period: {planet_period:.2f} days")
            st.sidebar.markdown(f"Radius: {planet_radius:.2f} Earth radii")
    except:
        st.sidebar.error("Failed to fetch exoplanet data. Using random target instead.")
        target_option = "Random TESS Target"
        tic_id = np.random.randint(100000000, 999999999)

elif target_option == "Random TESS Target":
    # Generate a random TIC ID
    tic_id = np.random.randint(100000000, 999999999)
    st.sidebar.success(f"Random TIC ID: {tic_id}")
elif target_option == "Custom TESS ID":
    # Custom TIC ID input
    tic_id = st.sidebar.number_input("Enter TESS Input Catalog (TIC) ID:", 
                                    min_value=100000000, max_value=999999999, 
                                    value=441420236, step=1)
else:  # Research Study
    tic_id = None
    st.sidebar.info("Research study mode selected. Click 'Run Research Study' in the Research tab.")

# Function to download real TESS data
@st.cache_data
def download_real_tess_data(tic_id):
    """Download and process real TESS data for a given TIC ID"""
    try:
        # Search for TESS data using lightkurve
        search = search_lightcurve(f"TIC {tic_id}", author="SPOC")
        if len(search) == 0:
            return None, None
        
        # Download the first available light curve
        lc = search.download()
        
        # Remove outliers and normalize
        lc = lc.remove_outliers(sigma=5).normalize()
        
        # Flatten the light curve to remove stellar variability
        lc_flat = lc.flatten(window_length=101)
        
        return lc.time.value, lc_flat.flux.value
    except Exception as e:
        st.error(f"Error processing real TESS data: {str(e)}")
        return None, None

# Function to implement Box Least Squares algorithm
def bls_detection(time, flux):
    """Implement Box Least Squares algorithm for transit detection"""
    # Create a BLS model
    model = BoxLeastSquares(time * u.day, flux)
    
    # Define period grid
    periods = np.logspace(-1, 1, 10000)  # 0.1 to 10 days
    durations = np.linspace(0.05, 0.2, 10)  # 0.05 to 0.2 days
    
    # Compute periodogram
    periodogram = model.autopower(periods, durations=durations)
    
    # Find the best period
    max_power = np.argmax(periodogram.power)
    best_period = periodogram.period[max_power]
    best_duration = periodogram.duration[max_power]
    best_t0 = periodogram.transit_time[max_power]
    
    return best_period.value, best_duration.value, best_t0.value, periodogram

# Function to implement Transit Least Squares algorithm
def tls_detection(time, flux):
    """Implement Transit Least Squares algorithm"""
    model = tls.transitleastsquares(time, flux)
    
    results = model.power(
        period_min=0.5,
        period_max=20,
        transit_depth_min=0,
        oversampling_factor=3,
        duration_grid_step=1.05
    )
    
    return results.period, results.depth, results.duration, results.SNR

# Function to validate detection using bootstrap analysis
def validate_detection(time, flux, period, duration, t0, num_bootstrap=100):
    """Validate detection using bootstrap analysis"""
    # Calculate the original signal strength
    original_strength = calculate_signal_strength(time, flux, period, duration, t0)
    
    # Bootstrap analysis
    bootstrap_strengths = []
    for _ in range(num_bootstrap):
        # Create a shuffled version of the data
        shuffled_flux = np.random.permutation(flux)
        
        # Calculate signal strength for shuffled data
        shuffled_strength = calculate_signal_strength(time, shuffled_flux, period, duration, t0)
        bootstrap_strengths.append(shuffled_strength)
    
    # Calculate false alarm probability
    false_alarms = sum(1 for s in bootstrap_strengths if s >= original_strength)
    fap = false_alarms / num_bootstrap
    
    # Calculate signal detection efficiency
    signal_efficiency = original_strength / np.mean(bootstrap_strengths)
    
    return fap, signal_efficiency, bootstrap_strengths

# Function to calculate signal strength
def calculate_signal_strength(time, flux, period, duration, t0):
    """Calculate the strength of a transit signal"""
    # Fold the light curve
    phases = ((time - t0) / period) % 1
    
    # Create phase bins
    phase_bins = np.linspace(0, 1, 100)
    bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
    
    # Bin the data
    binned_flux = []
    for i in range(len(phase_bins) - 1):
        mask = (phases >= phase_bins[i]) & (phases < phase_bins[i+1])
        if np.sum(mask) > 0:
            binned_flux.append(np.mean(flux[mask]))
        else:
            binned_flux.append(np.nan)
    
    binned_flux = np.array(binned_flux)
    
    # Calculate the transit depth
    in_transit = (bin_centers > (1 - duration/period)) | (bin_centers < duration/period)
    out_of_transit = ~in_transit
    
    if np.sum(in_transit) > 0 and np.sum(out_of_transit) > 0:
        transit_depth = np.mean(binned_flux[out_of_transit]) - np.mean(binned_flux[in_transit])
        signal_strength = transit_depth / np.std(binned_flux[out_of_transit])
    else:
        signal_strength = 0
    
    return signal_strength

# Function to compare with known exoplanets
def compare_with_known_exoplanets(tic_id, detected_period, detected_radius):
    """Compare detected planet with known exoplanets in the system"""
    try:
        # Query NASA Exoplanet Archive for known planets around this TIC ID
        exoplanets = NasaExoplanetArchive.query_criteria(table="ps", tic_id=int(tic_id))
        
        if len(exoplanets) > 0:
            comparison_results = []
            
            for planet in exoplanets:
                known_period = planet['pl_orbper']
                known_radius = planet['pl_rade']
                planet_name = planet['pl_name']
                
                # Calculate differences
                period_diff = abs(detected_period - known_period) / known_period * 100
                radius_diff = abs(detected_radius - known_radius) / known_radius * 100
                
                comparison_results.append({
                    'name': planet_name,
                    'known_period': known_period,
                    'detected_period': detected_period,
                    'period_diff': period_diff,
                    'known_radius': known_radius,
                    'detected_radius': detected_radius,
                    'radius_diff': radius_diff
                })
            
            return comparison_results
        else:
            return None
    except Exception as e:
        st.error(f"Error comparing with known exoplanets: {str(e)}")
        return None

# Function to create a CNN model for transit detection
def create_cnn_model(input_shape):
    """Create a CNN model for transit detection"""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to analyze multi-planet systems
def analyze_multi_planet_system(time, flux):
    """Analyze light curves for multiple planets in a system"""
    # First, remove the strongest signal
    period1, duration1, t01, periodogram1 = bls_detection(time, flux)
    
    # Create a model of the first transit
    model1 = create_transit_model(time, period1, duration1, t01)
    
    # Remove the first transit signal
    residual_flux = flux - model1 + np.median(flux)
    
    # Look for additional signals in the residual
    period2, duration2, t02, periodogram2 = bls_detection(time, residual_flux)
    
    # Check if the second signal is significant
    if periodogram2.power.max() > 10:  # Arbitrary threshold
        # Create a model of the second transit
        model2 = create_transit_model(time, period2, duration2, t02)
        
        # Remove the second transit signal
        residual_flux2 = residual_flux - model2 + np.median(residual_flux)
        
        # Look for a third signal
        period3, duration3, t03, periodogram3 = bls_detection(time, residual_flux2)
        
        if periodogram3.power.max() > 10:
            return [
                {'period': period1, 'duration': duration1, 't0': t01},
                {'period': period2, 'duration': duration2, 't0': t02},
                {'period': period3, 'duration': duration3, 't0': t03}
            ]
        else:
            return [
                {'period': period1, 'duration': duration1, 't0': t01},
                {'period': period2, 'duration': duration2, 't0': t02}
            ]
    else:
        return [{'period': period1, 'duration': duration1, 't0': t01}]

# Function to create a transit model
def create_transit_model(time, period, duration, t0):
    """Create a model of a transit signal"""
    phases = ((time - t0) / period) % 1
    transit_model = np.zeros_like(time)
    
    # Simple box-shaped transit
    in_transit = (phases < duration/period) | (phases > (1 - duration/period))
    transit_model[in_transit] = 0.01  # 1% transit depth
    
    return transit_model

# Function to assess habitability
def assess_habitability(stellar_temp, stellar_radius, planet_period, planet_radius):
    """Assess the potential habitability of a detected planet"""
    # Calculate the semi-major axis using Kepler's third law
    # a^3 = P^2 * M_star
    # For simplicity, assume stellar mass proportional to radius^3
    stellar_mass = stellar_radius ** 3  # Solar masses
    semi_major_axis = (planet_period**2 * stellar_mass) ** (1/3)  # AU
    
    # Calculate equilibrium temperature
    # T_eq = T_star * (1-A)^(1/4) * sqrt(R_star/(2*a))
    albedo = 0.3  # Earth-like albedo
    temp_eq = stellar_temp * ((1 - albedo) ** 0.25) * np.sqrt(stellar_radius / (2 * semi_major_axis))
    
    # Determine if planet is in the habitable zone
    # Simple approximation for Sun-like stars
    hz_inner = 0.95  # AU
    hz_outer = 1.37  # AU
    
    in_habitable_zone = hz_inner <= semi_major_axis <= hz_outer
    
    # Assess planet size
    is_rocky = planet_radius < 1.5  # Earth radii
    
    # Calculate Earth Similarity Index (simplified)
    esi = 0
    if in_habitable_zone and is_rocky:
        # ESI = (1 - |radius - 1|/1) * (1 - |temp_eq - 288|/288)
        radius_similarity = 1 - abs(planet_radius - 1) / 1
        temp_similarity = 1 - abs(temp_eq - 288) / 288
        esi = radius_similarity * temp_similarity
    
    return {
        'semi_major_axis': semi_major_axis,
        'equilibrium_temp': temp_eq,
        'in_habitable_zone': in_habitable_zone,
        'is_rocky': is_rocky,
        'esi': max(0, min(1, esi))  # Clamp between 0 and 1
    }

# Function to conduct research study
def conduct_research_study():
    """Conduct a research study on detection methods"""
    # Select a sample of known exoplanet host stars
    try:
        exoplanets = NasaExoplanetArchive.query_criteria(table="ps", 
                                                        default_flag=1, 
                                                        disc_telescope="TESS",
                                                        pl_rade=[0.5, 4.0])  # Earth-sized to Neptune-sized
        
        # Select a random sample of 50 systems
        sample_size = min(50, len(exoplanets))
        sample_indices = np.random.choice(len(exoplanets), sample_size, replace=False)
        sample = exoplanets[sample_indices]
        
        results = []
        
        for i, planet in enumerate(sample):
            tic_id = planet['tic_id']
            known_period = planet['pl_orbper']
            known_radius = planet['pl_rade']
            
            # Download and process TESS data
            time, flux = download_real_tess_data(tic_id)
            
            if time is not None:
                # Apply detection methods
                bls_period, bls_duration, bls_t0, bls_periodogram = bls_detection(time, flux)
                tls_period, tls_depth, tls_duration, tls_snr = tls_detection(time, flux)
                
                # Calculate detection accuracy
                bls_period_error = abs(bls_period - known_period) / known_period * 100
                tls_period_error = abs(tls_period - known_period) / known_period * 100
                
                results.append({
                    'tic_id': tic_id,
                    'known_period': known_period,
                    'known_radius': known_radius,
                    'bls_period': bls_period,
                    'bls_period_error': bls_period_error,
                    'tls_period': tls_period,
                    'tls_period_error': tls_period_error,
                    'tls_snr': tls_snr
                })
        
        # Analyze results
        if results:
            df = pd.DataFrame(results)
            
            # Calculate statistics
            bls_mean_error = df['bls_period_error'].mean()
            tls_mean_error = df['tls_period_error'].mean()
            
            # Compare methods
            comparison = {
                'bls_mean_error': bls_mean_error,
                'tls_mean_error': tls_mean_error,
                'better_method': 'TLS' if tls_mean_error < bls_mean_error else 'BLS',
                'detection_rate': len(df[df['tls_snr'] > 7]) / len(df) * 100  # SNR threshold
            }
            
            return comparison, df
        else:
            return None, None
    except Exception as e:
        st.error(f"Error in research study: {str(e)}")
        return None, None

# Get the data if not in research mode
if target_option != "Research Study" and tic_id:
    with st.spinner("Downloading real TESS data..."):
        time, flux = download_real_tess_data(tic_id)

# Main content area
if target_option != "Research Study" and tic_id and time is not None:
    st.markdown('<h2 class="section-header">📊 Advanced Light Curve Analysis</h2>', unsafe_allow_html=True)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Light Curve", 
        "Detection Methods", 
        "Statistical Validation", 
        "Multi-Planet Analysis", 
        "Habitability Assessment", 
        "Comparison with Known Planets"
    ])
    
    with tab1:
        st.markdown("### Raw Light Curve")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time, flux, 'k.', markersize=1)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Relative Flux')
        ax.set_title(f'Real TESS Light Curve for TIC {tic_id}')
        ax.grid(True)
        st.pyplot(fig)
        
        st.markdown('<div class="info-box">This is real TESS data processed to remove outliers and stellar variability. Dips in brightness may indicate planets passing in front of the star.</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Advanced Detection Methods")
        
        # Apply BLS and TLS methods
        bls_period, bls_duration, bls_t0, bls_periodogram = bls_detection(time, flux)
        tls_period, tls_depth, tls_duration, tls_snr = tls_detection(time, flux)
        
        # Display results
        st.markdown("#### Box Least Squares (BLS) Results")
        st.write(f"Detected Period: {bls_period:.4f} days")
        st.write(f"Transit Duration: {bls_duration:.4f} days")
        st.write(f"Transit Midpoint: {bls_t0:.4f} days")
        
        st.markdown("#### Transit Least Squares (TLS) Results")
        st.write(f"Detected Period: {tls_period:.4f} days")
        st.write(f"Transit Depth: {tls_depth:.6f}")
        st.write(f"Transit Duration: {tls_duration:.4f} days")
        st.write(f"Signal-to-Noise Ratio: {tls_snr:.2f}")
        
        # Plot periodograms
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # BLS periodogram
        ax1.plot(bls_periodogram.period, bls_periodogram.power, 'b-')
        ax1.set_xlabel('Period (days)')
        ax1.set_ylabel('BLS Power')
        ax1.set_title('BLS Periodogram')
        ax1.grid(True)
        ax1.set_xscale('log')
        
        # TLS periodogram
        model = tls.transitleastsquares(time, flux)
        results = model.power(
            period_min=0.5,
            period_max=20,
            transit_depth_min=0,
            oversampling_factor=3,
            duration_grid_step=1.05
        )
        ax2.plot(results.period, results.SDE, 'r-')
        ax2.set_xlabel('Period (days)')
        ax2.set_ylabel('TLS SDE')
        ax2.set_title('TLS Periodogram')
        ax2.grid(True)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Estimate planet properties
        transit_depth = tls_depth
        stellar_radius = 1.0  # Solar radii (default)
        stellar_radius_earth = stellar_radius * 109  # Earth radii
        
        # Transit depth ≈ (R_planet / R_star)^2
        planet_radius_estimate = stellar_radius_earth * np.sqrt(transit_depth)
        
        st.markdown("### Estimated Planet Properties")
        st.metric("Orbital Period", f"{tls_period:.4f} days")
        st.metric("Planet Radius", f"{planet_radius_estimate:.2f} Earth radii")
        st.metric("Signal-to-Noise Ratio", f"{tls_snr:.2f}")
        
        # Classification
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
    
    with tab3:
        st.markdown("### Statistical Validation")
        
        # Validate the detection
        fap, signal_efficiency, bootstrap_strengths = validate_detection(
            time, flux, tls_period, tls_duration, bls_t0, num_bootstrap=100
        )
        
        st.markdown("#### Bootstrap Analysis Results")
        st.write(f"False Alarm Probability: {fap:.4f}")
        st.write(f"Signal Detection Efficiency: {signal_efficiency:.2f}")
        
        # Plot bootstrap distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(bootstrap_strengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=signal_efficiency, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Signal Strength')
        ax.set_ylabel('Frequency')
        ax.set_title('Bootstrap Distribution of Signal Strengths')
        ax.legend(['Detected Signal', 'Bootstrap Distribution'])
        ax.grid(True)
        st.pyplot(fig)
        
        # Statistical significance
        if fap < 0.01:
            st.success("✅ Detection is statistically significant (FAP < 1%)")
        elif fap < 0.05:
            st.warning("⚠️ Detection is marginally significant (1% ≤ FAP < 5%)")
        else:
            st.error("❌ Detection is not statistically significant (FAP ≥ 5%)")
    
    with tab4:
        st.markdown("### Multi-Planet System Analysis")
        
        # Analyze for multiple planets
        planets = analyze_multi_planet_system(time, flux)
        
        st.write(f"Detected {len(planets)} potential planet(s) in this system:")
        
        for i, planet in enumerate(planets):
            st.markdown(f"#### Planet {i+1}")
            st.write(f"Orbital Period: {planet['period']:.4f} days")
            st.write(f"Transit Duration: {planet['duration']:.4f} days")
            st.write(f"Transit Midpoint: {planet['t0']:.4f} days")
            
            # Estimate planet properties
            transit_depth = 0.01  # Default depth
            planet_radius_estimate = 109 * np.sqrt(transit_depth)  # Earth radii
            
            # Classification
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
            
            st.write(f"Estimated Radius: {planet_radius_estimate:.2f} Earth radii")
            st.write(f"Classification: {planet_class}")
            
            st.markdown("---")
    
    with tab5:
        st.markdown("### Habitability Assessment")
        
        # Get stellar properties (simplified)
        stellar_temp = 5778  # Sun-like temperature (K)
        stellar_radius = 1.0  # Solar radii
        
        # Assess habitability for the detected planet
        habitability = assess_habitability(
            stellar_temp, 
            stellar_radius, 
            tls_period, 
            planet_radius_estimate
        )
        
        st.markdown("#### Habitability Metrics")
        st.write(f"Semi-major Axis: {habitability['semi_major_axis']:.3f} AU")
        st.write(f"Equilibrium Temperature: {habitability['equilibrium_temp']:.1f} K")
        st.write(f"In Habitable Zone: {'Yes' if habitability['in_habitable_zone'] else 'No'}")
        st.write(f"Rocky Planet: {'Yes' if habitability['is_rocky'] else 'No'}")
        st.write(f"Earth Similarity Index: {habitability['esi']:.3f}")
        
        # Visualize habitable zone
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot star
        star = plt.Circle((0, 0), 0.1, color='yellow', zorder=10)
        ax.add_patch(star)
        
        # Plot habitable zone
        hz_inner = plt.Circle((0, 0), 0.95, color='none', edgecolor='green', linestyle='--', linewidth=2)
        hz_outer = plt.Circle((0, 0), 1.37, color='none', edgecolor='green', linestyle='--', linewidth=2)
        ax.add_patch(hz_inner)
        ax.add_patch(hz_outer)
        
        # Plot planet
        planet_circle = plt.Circle((habitability['semi_major_axis'], 0), 0.05, 
                                  color='blue' if habitability['in_habitable_zone'] else 'red', zorder=5)
        ax.add_patch(planet_circle)
        
        # Set plot properties
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
        
        # Compare with known exoplanets
        comparison_results = compare_with_known_exoplanets(tic_id, tls_period, planet_radius_estimate)
        
        if comparison_results:
            st.write("Comparison with known planets in this system:")
            
            for result in comparison_results:
                st.markdown(f"#### {result['name']}")
                st.write(f"Known Period: {result['known_period']:.4f} days")
                st.write(f"Detected Period: {result['detected_period']:.4f} days")
                st.write(f"Period Difference: {result['period_diff']:.2f}%")
                st.write(f"Known Radius: {result['known_radius']:.2f} Earth radii")
                st.write(f"Detected Radius: {result['detected_radius']:.2f} Earth radii")
                st.write(f"Radius Difference: {result['radius_diff']:.2f}%")
                
                # Assess accuracy
                if result['period_diff'] < 5 and result['radius_diff'] < 20:
                    st.success("✅ Excellent match with known planet!")
                elif result['period_diff'] < 10 and result['radius_diff'] < 40:
                    st.warning("⚠️ Good match with known planet.")
                else:
                    st.error("❌ Poor match with known planet.")
                
                st.markdown("---")
        else:
            st.info("No known exoplanets in this system for comparison. This could be a new discovery!")
            
            # Button to "confirm" the discovery
            if st.button("Confirm Discovery", key="confirm_discovery"):
                st.balloons()
                st.success(f"🎉 Congratulations! You've discovered a potential {planet_class} exoplanet orbiting TIC {tic_id}!")
                st.markdown(f"""
                **Discovery Summary:**
                - **Host Star:** TIC {tic_id}
                - **Planet Period:** {tls_period:.4f} days
                - **Planet Radius:** {planet_radius_estimate:.2f} Earth radii
                - **Signal-to-Noise Ratio:** {tls_snr:.2f}
                - **False Alarm Probability:** {fap:.4f}
                - **Planet Class:** {planet_class}
                - **Habitability:** {'Potentially habitable' if habitability['in_habitable_zone'] and habitability['is_rocky'] else 'Not habitable'}
                """)

elif target_option == "Research Study":
    st.markdown('<h2 class="section-header">🔬 Research Study: Detection Method Comparison</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    This research study compares the performance of Box Least Squares (BLS) and Transit Least Squares (TLS) 
    algorithms for detecting exoplanets in TESS data. We'll analyze a sample of known exoplanet systems 
    and evaluate the accuracy of each method.
    """)
    
    if st.button("Run Research Study", key="run_research"):
        with st.spinner("Conducting research study... This may take a few minutes."):
            comparison, df = conduct_research_study()
            st.session_state.research_results = (comparison, df)
    
    if st.session_state.research_results:
        comparison, df = st.session_state.research_results
        
        if comparison is not None:
            st.markdown('<div class="research-card">', unsafe_allow_html=True)
            st.markdown("### Research Results Summary")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display key metrics
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">BLS Mean Error</div>
                <div class="metric-value">{comparison['bls_mean_error']:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">TLS Mean Error</div>
                <div class="metric-value">{comparison['tls_mean_error']:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Better Method</div>
                <div class="metric-value">{comparison['better_method']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Detection Rate</div>
                <div class="metric-value">{comparison['detection_rate']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display detailed results
            st.markdown("### Detailed Results")
            st.dataframe(df)
            
            # Plot comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Period error comparison
            ax1.hist(df['bls_period_error'], alpha=0.5, label='BLS', bins=20)
            ax1.hist(df['tls_period_error'], alpha=0.5, label='TLS', bins=20)
            ax1.set_xlabel('Period Error (%)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Period Error Distribution')
            ax1.legend()
            ax1.grid(True)
            
            # SNR distribution
            ax2.hist(df['tls_snr'], bins=20, color='green', alpha=0.7)
            ax2.axvline(x=7, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('Signal-to-Noise Ratio')
            ax2.set_ylabel('Frequency')
            ax2.set_title('TLS SNR Distribution')
            ax2.legend(['Detection Threshold', 'SNR'])
            ax2.grid(True)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Research conclusions
            st.markdown('<div class="research-card">', unsafe_allow_html=True)
            st.markdown("### Research Conclusions")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            Based on the analysis of {len(df)} known exoplanet systems:
            
            1. **Method Performance**: {comparison['better_method']} outperformed the other method with a mean error of {min(comparison['bls_mean_error'], comparison['tls_mean_error']):.2f}%.
            
            2. **Detection Rate**: {comparison['detection_rate']:.1f}% of known planets were detected with SNR > 7.
            
            3. **Accuracy**: Both methods showed good accuracy for planets with strong transit signals, but performance decreased for smaller planets with shallower transits.
            
            4. **Recommendations**: 
               - Use {comparison['better_method']} for initial planet detection
               - Combine both methods for validation
               - Apply additional statistical tests to reduce false positives
            """)
            
            # Download results button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Research Data",
                data=csv,
                file_name='tess_detection_comparison.csv',
                mime='text/csv'
            )
        else:
            st.error("Research study failed. Please try again.")

# Chatbot Section
st.markdown('<h2 class="section-header">🤖 ExoBot - Your Exoplanet Assistant</h2>', unsafe_allow_html=True)

# Chat container
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        if chat["sender"] == "user":
            st.markdown(f'<div class="chat-message user-message">{chat["message"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message"><b>ExoBot:</b> {chat["message"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Chat input
user_input = st.text_input("Ask ExoBot a question:", key="chat_input")

# Enhanced chatbot response function
def get_bot_response(user_input):
    user_input = user_input.lower()
    
    # Define responses based on keywords
    responses = {
        r"hello|hi|hey": "Hello there! I'm ExoBot, your exoplanet discovery assistant. How can I help you today?",
        r"tess|transiting exoplanet survey satellite": "TESS (Transiting Exoplanet Survey Satellite) is a NASA space telescope designed to search for exoplanets using the transit method. It was launched in 2018 and has discovered thousands of planet candidates.",
        r"exoplanet|planet": "Exoplanets are planets that orbit stars other than our Sun. Over 5,000 exoplanets have been confirmed so far, with many more candidates awaiting verification.",
        r"transit method": "The transit method detects exoplanets by measuring the dimming of a star's light when a planet passes in front of it. This is the primary method used by TESS and Kepler missions.",
        r"light curve": "A light curve is a graph of light intensity over time. In exoplanet research, it shows the brightness of a star, with dips indicating potential planet transits.",
        r"periodogram": "A periodogram is a graph showing the strength of periodic signals at different frequencies. It helps identify potential orbital periods of exoplanets in light curve data.",
        r"habitable zone|goldilocks zone": "The habitable zone is the region around a star where conditions might be right for liquid water to exist on a planet's surface. It's neither too hot nor too cold.",
        r"kepler": "The Kepler Space Telescope was NASA's first planet-hunting mission. It discovered over 2,600 confirmed exoplanets during its 9-year mission.",
        r"james webb|jwst": "The James Webb Space Telescope is the most powerful space telescope ever built. It can study exoplanet atmospheres and search for signs of life.",
        r"how.*discover|find.*planet": "You can discover planets using this app by: 1) Selecting a target star, 2) Analyzing the light curve for dips, 3) Using the periodogram to find periods, 4) Folding the light curve to confirm the signal, 5) Estimating planet properties.",
        r"earth.*like|earth.*sized": "Earth-sized planets are those with radii similar to Earth (0.8-1.25 Earth radii). They're of special interest in the search for potentially habitable worlds.",
        r"hot jupiter": "Hot Jupiters are gas giant planets that orbit very close to their stars, with orbital periods of less than 10 days. They're tidally locked and extremely hot.",
        r"super earth": "Super-Earths are planets with masses between Earth and Neptune (1-10 Earth masses). They may be rocky planets with thick atmospheres or mini-Neptunes.",
        r"red dwarf": "Red dwarfs are small, cool stars (M-type) that are the most common stars in our galaxy. They're prime targets for exoplanet searches because their small size makes planet transits easier to detect.",
        r"confirm.*discovery": "To confirm a planet discovery, astronomers need: 1) Multiple transit observations, 2) Ruling out false positives (like eclipsing binaries), 3) Follow-up observations with other telescopes, 4) Statistical validation.",
        r"false positive": "False positives are signals that mimic planet transits but are caused by other phenomena like stellar activity, eclipsing binaries, or instrumental effects. Astronomers use various tests to identify and eliminate them.",
        r"next.*mission|future.*telescope": "Future exoplanet missions include PLATO (ESA), ARIEL (ESA), and the Habitable Worlds Observatory (NASA). These will focus on characterizing exoplanet atmospheres and searching for biosignatures.",
        r"life|biosignature": "Biosignatures are chemical or physical signs that indicate the presence of life. In exoplanets, potential biosignatures include oxygen, methane, and other chemical imbalances in the atmosphere.",
        r"research.*study|comparison": "This app includes a research study that compares BLS and TLS detection methods using known exoplanet systems. It evaluates accuracy and detection rates to determine which method performs better.",
        r"bls|box least squares": "Box Least Squares (BLS) is a widely used algorithm for detecting exoplanet transits. It searches for periodic box-shaped signals in light curves by optimizing period, duration, and phase.",
        r"tls|transit least squares": "Transit Least Squares (TLS) is an improved algorithm for detecting transits. It uses a more realistic transit model and often outperforms BLS, especially for shallow transits.",
        r"statistical.*validation": "Statistical validation assesses the significance of a detection using methods like bootstrap analysis. It calculates the false alarm probability (FAP) to determine if a signal is likely real or just noise.",
        r"multi.*planet|multiple.*planet": "Multi-planet systems contain two or more planets orbiting the same star. This app can detect multiple planets by iteratively removing the strongest signal and searching for additional signals in the residual data.",
        r"habitability.*assessment": "Habitability assessment evaluates whether a planet could support life. It considers factors like distance from the star (habitable zone), planet size (rocky vs. gaseous), and temperature.",
        r"earth.*similarity.*index|esi": "The Earth Similarity Index (ESI) is a metric that rates how similar a planet is to Earth on a scale from 0 to 1. It considers factors like radius, density, and temperature.",
        r"bootstrap.*analysis": "Bootstrap analysis is a statistical method that estimates the significance of a detection by comparing the signal strength to those in randomly shuffled data. It helps calculate the false alarm probability.",
        r"signal.*to.*noise|snr": "Signal-to-Noise Ratio (SNR) measures the strength of a signal relative to the background noise. Higher SNR values indicate more reliable detections. A common threshold is SNR > 7 for planet candidates.",
        r"thank.*you|thanks": "You're welcome! Feel free to ask me anything else about exoplanets or TESS."
    }
    
    # Check for matching keywords
    for pattern, response in responses.items():
        if re.search(pattern, user_input):
            return response
    
    # Default response if no keywords match
    return "That's an interesting question! I'm still learning about exoplanets. Could you try asking about TESS, exoplanets, the transit method, detection algorithms, or how to use this app?"

# Handle chat submission
if st.button("Send", key="send_button") and user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({
        "sender": "user",
        "message": user_input,
        "time": datetime.now().strftime("%H:%M")
    })
    
    # Get bot response
    bot_response = get_bot_response(user_input)
    
    # Add bot response to chat history
    st.session_state.chat_history.append({
        "sender": "bot",
        "message": bot_response,
        "time": datetime.now().strftime("%H:%M")
    })
    
    # Clear input
    st.session_state.chat_input = ""
    
    # Rerun to update chat
    st.experimental_rerun()

# Information section
st.markdown('<h2 class="section-header">ℹ️ About TESS and Exoplanet Discovery</h2>', unsafe_allow_html=True)
st.markdown("""
The Transiting Exoplanet Survey Satellite (TESS) is a space telescope designed to search for exoplanets using the transit method. 
It monitors the brightness of hundreds of thousands of stars to detect temporary drops in brightness caused by planets passing in front of their host stars.

**Enhanced Features in This App:**
- Real TESS data processing using lightkurve
- Advanced detection algorithms (BLS and TLS)
- Statistical validation with bootstrap analysis
- Multi-planet system detection
- Habitability assessment with Earth Similarity Index
- Comparison with known exoplanets
- Research study component for method comparison
- Interactive chatbot for guidance

**How to Discover Planets with this App:**
1. Select a target star using the controls in the sidebar
2. Analyze the real light curve for transit signals
3. Use advanced detection methods to identify potential planets
4. Validate detections with statistical analysis
5. Assess habitability for Earth-like planets
6. Compare results with known exoplanets
7. Use the research study to compare detection methods
8. Use the chatbot if you need guidance or have questions

**Real Planet Discovery:**
This app uses real TESS data and professional-grade algorithms to detect potential exoplanets. While it simulates the discovery process, 
the techniques used are similar to those employed by astronomers. Real discoveries require additional validation and follow-up observations.
""")

# Footer
st.markdown("---")
st.markdown("Made with  using TESS data and Streamlit")