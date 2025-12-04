
import os
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import from main
import main
from main import (
    download_brdc_with_fallback, 
    load_rinex_dir, 
    SkyImage, 
    Satellite, 
    propagate_rk4, 
    _gps_ecef_pos_vel, 
    _ecef_to_eci,
    RINEX_DIR
)

# Configuration
START_DATE = datetime.date(2025, 11, 20)
DURATION_DAYS = 14
ANALYSIS_DIR = "analysis_data"
TRUTH_DIR = os.path.join(ANALYSIS_DIR, "truth")
START_DIR = os.path.join(ANALYSIS_DIR, "start")

def ensure_data():
    """Download RINEX files for the analysis period."""
    os.makedirs(TRUTH_DIR, exist_ok=True)
    os.makedirs(START_DIR, exist_ok=True)

    # Download Start Day (Day 0)
    print(f"Downloading Start Day: {START_DATE}")
    try:
        download_brdc_with_fallback(START_DATE, out_dir=START_DIR)
    except Exception as e:
        print(f"Failed to download start day: {e}")
        return False

    # Download Truth Days (Day 0 to Day 14)
    # We include Day 0 in truth to verify t=0 accuracy (should be near zero)
    for i in range(DURATION_DAYS + 1):
        day = START_DATE + datetime.timedelta(days=i)
        print(f"Downloading Truth Day {i}: {day}")
        try:
            download_brdc_with_fallback(day, out_dir=TRUTH_DIR)
        except Exception as e:
            print(f"Warning: Failed to download truth for {day}: {e}")
            # Continue, maybe we have gaps
            
    return True

def get_truth_position(sat: Satellite, t: datetime.datetime):
    """
    Calculate the 'truth' position of a satellite at time t using the best available broadcast parameters.
    """
    # Find parameters with Toe closest to t
    best_params = None
    min_diff = float('inf')
    
    # We search all entries in the satellite
    # Optimization: entries are sorted by epoch. We could binary search.
    # But linear scan is fine for ~14 days * ~12 entries/day = ~168 entries.
    
    for epoch, params in sat.entries:
        # Toe is usually the epoch key in our parser
        diff = abs((t - epoch).total_seconds())
        if diff < min_diff:
            min_diff = diff
            best_params = (epoch, params)
            
    if not best_params:
        return None
        
    epoch_oe, p = best_params
    
    # Check validity? Broadcast usually valid for +/- 2 hours (GPS).
    # But for analysis we stretch it if needed, or just take the closest.
    # If diff is too large (e.g. > 4 hours), maybe we shouldn't trust it.
    if min_diff > 4 * 3600:
        return None

    sys = sat.id[0]
    
    try:
        if sys in ('R', 'S'):
            # GLONASS/SBAS: State is given at Toe. We need to propagate?
            # GLONASS parameters are state vector at Toe.
            # Simple extrapolation: r = r0 + v0*dt + ...
            # But main.py's _ecef_to_eci just converts.
            # To get position at t != Toe, we technically need to integrate GLONASS equations.
            # Our propagator does that!
            # So "truth" for GLONASS is hard to get analytically from a single message at t != Toe.
            # We will skip GLONASS for this accuracy analysis and focus on GPS/Galileo/BeiDou 
            # which use Keplerian-like expansions valid over time.
            return None
            
        # GPS/Galileo/BeiDou
        # Evaluate position at time t
        xE, yE, zE, vxE, vyE, vzE = _gps_ecef_pos_vel(t, p)
        
        # Convert to ECI
        # Note: _ecef_to_eci uses OMEGA_E.
        x, y, z, vx, vy, vz = _ecef_to_eci(t, xE, yE, zE, vxE, vyE, vzE)
        return (x, y, z)
        
    except Exception:
        return None

def run_analysis():
    if not ensure_data():
        print("Aborting analysis due to missing data.")
        return

    print("Loading Start Data...")
    start_data = load_rinex_dir(START_DIR)
    start_sats = {sid: Satellite(sid, entries) for sid, entries in start_data.items()}
    sky_image = SkyImage(start_sats)
    
    print("Loading Truth Data...")
    truth_data = load_rinex_dir(TRUTH_DIR)
    # Merge all truth data into one Satellite object per ID
    truth_sats = {}
    for sid, entries in truth_data.items():
        if sid not in truth_sats:
            truth_sats[sid] = Satellite(sid, entries)
        else:
            # Merge entries
            existing = truth_sats[sid].entries
            # Add new ones, avoiding duplicates
            existing_epochs = set(e for e, _ in existing)
            for e, p in entries:
                if e not in existing_epochs:
                    existing.append((e, p))
            truth_sats[sid].entries.sort(key=lambda x: x[0])

    print(f"Propagating for {DURATION_DAYS} days...")
    # Propagate
    # We want high resolution for the plot? e.g. every 1 hour.
    sky_image.propagate_all(
        days=DURATION_DAYS,
        output_every_minutes=60,
        step_seconds=60.0,
        forces=['central', 'J2', 'J3', 'J4', 'Sun', 'Moon', 'SRP']
    )
    
    print("Calculating Errors...")
    # Time vs RMS Error
    time_points = []
    rms_errors = []
    
    # Collect all prediction times
    # Assuming all sats have same steps roughly
    sample_sid = next(iter(sky_image.predictions))
    sample_preds = sky_image.predictions[sample_sid]
    
    # We iterate over the time steps in the prediction
    for t, _ in sample_preds:
        errors = []
        for sid, pred_list in sky_image.predictions.items():
            # Find prediction for this sat at time t
            # (Optimization: pred_list is sorted, we could track index)
            pred = next((state for pt, state in pred_list if pt == t), None)
            if not pred:
                continue
                
            # Find truth
            if sid not in truth_sats:
                continue
            
            # Skip GLONASS for now (see get_truth_position)
            if sid.startswith('R') or sid.startswith('S'):
                continue
                
            truth_pos = get_truth_position(truth_sats[sid], t)
            if not truth_pos:
                continue
                
            # Calc distance
            px, py, pz = pred[:3]
            tx, ty, tz = truth_pos
            dist = math.sqrt((px-tx)**2 + (py-ty)**2 + (pz-tz)**2)
            
            # Sanity check: if error > 1000km, maybe matching wrong epoch or invalid truth
            if dist < 500000: # 500km
                errors.append(dist)
        
        if errors:
            rms = np.sqrt(np.mean(np.array(errors)**2))
            # Convert to days from start
            dt_days = (t - sample_preds[0][0]).total_seconds() / 86400.0
            time_points.append(dt_days)
            rms_errors.append(rms / 1000.0) # Convert to km
            
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, rms_errors, label='RMS Position Error')
    plt.xlabel('Propagation Time (days)')
    plt.ylabel('Error (km)')
    plt.title(f'Propagation Accuracy vs Time (Start: {START_DATE})')
    plt.grid(True)
    plt.legend()
    
    out_file = "static/accuracy_plot.png"
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    run_analysis()
