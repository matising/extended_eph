
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
CALIBRATION_DAYS = 3
ANALYSIS_DIR = "analysis_data"
TRUTH_DIR = os.path.join(ANALYSIS_DIR, "truth")
START_DIR = os.path.join(ANALYSIS_DIR, "start")
CALIB_DIR = os.path.join(ANALYSIS_DIR, "calib")

def ensure_data():
    """Download RINEX files for the analysis period."""
    os.makedirs(TRUTH_DIR, exist_ok=True)
    os.makedirs(START_DIR, exist_ok=True)
    os.makedirs(CALIB_DIR, exist_ok=True)

    # Download Start Day (Day 0)
    print(f"Downloading Start Day: {START_DATE}")
    try:
        download_brdc_with_fallback(START_DATE, out_dir=START_DIR)
    except Exception as e:
        print(f"Failed to download start day: {e}")
        return False

    # Download Calibration Days (Day -3 to Day -1)
    for i in range(1, CALIBRATION_DAYS + 1):
        day = START_DATE - datetime.timedelta(days=i)
        print(f"Downloading Calibration Day -{i}: {day}")
        try:
            download_brdc_with_fallback(day, out_dir=CALIB_DIR)
        except Exception as e:
            print(f"Warning: Failed to download calib for {day}: {e}")

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

    # 1. Load Data
    print("Loading Start Data (Day 0)...")
    start_data = load_rinex_dir(START_DIR)
    start_sats = {sid: Satellite(sid, entries) for sid, entries in start_data.items()}
    
    print("Loading Calibration Data (Day -3 to -1)...")
    calib_data = load_rinex_dir(CALIB_DIR)
    # Merge calib data
    calib_sats = {}
    for sid, entries in calib_data.items():
        if sid not in calib_sats:
            calib_sats[sid] = Satellite(sid, entries)
        else:
            existing = calib_sats[sid].entries
            existing_epochs = set(e for e, _ in existing)
            for e, p in entries:
                if e not in existing_epochs:
                    existing.append((e, p))
            calib_sats[sid].entries.sort(key=lambda x: x[0])
            
    # Also merge Start Data into Calib Sats because calibration range might extend to Day 0
    for sid, sat in start_sats.items():
        if sid in calib_sats:
            existing = calib_sats[sid].entries
            existing_epochs = set(e for e, _ in existing)
            for e, p in sat.entries:
                if e not in existing_epochs:
                    existing.append((e, p))
            calib_sats[sid].entries.sort(key=lambda x: x[0])
        else:
            calib_sats[sid] = Satellite(sid, sat.entries)

    print("Loading Truth Data...")
    truth_data = load_rinex_dir(TRUTH_DIR)
    truth_sats = {}
    for sid, entries in truth_data.items():
        if sid not in truth_sats:
            truth_sats[sid] = Satellite(sid, entries)
        else:
            existing = truth_sats[sid].entries
            existing_epochs = set(e for e, _ in existing)
            for e, p in entries:
                if e not in existing_epochs:
                    existing.append((e, p))
            truth_sats[sid].entries.sort(key=lambda x: x[0])

    # 2. Uncalibrated Propagation (Start Day 0)
    print(f"Running Uncalibrated Propagation ({DURATION_DAYS} days)...")
    sky_image_uncal = SkyImage(start_sats)
    sky_image_uncal.propagate_all(
        days=DURATION_DAYS,
        output_every_minutes=60,
        step_seconds=60.0,
        forces=['central', 'J2', 'J3', 'J4', 'Sun', 'Moon', 'SRP']
    )

    # 3. Calibrated Propagation (Start Day -3, fit to Day 0)
    print(f"Running Calibration & Propagation...")
    # sky_image_cal = SkyImage(calib_sats) # Not needed, we do manual propagation
    
    # We need to manually calibrate and propagate because SkyImage.propagate_all doesn't do calibration
    # We will mimic propagate_all but with calibration
    
    calibrated_predictions = {}
    
    # Filter to same sats as uncalibrated for fair comparison (or all available)
    # Let's use intersection
    common_sids = set(start_sats.keys()) & set(calib_sats.keys())
    
    for sid in common_sids:
        sat = calib_sats[sid]
        epochs = sat.epochs()
        if not epochs: 
            print(f"DEBUG: No epochs for {sid}")
            continue
        
        # Define calibration range: from first available in calib set up to Day 0 start
        # Actually, we want to use all data up to START_DATE
        # Find index of first epoch >= START_DATE
        idx_start_forecast = -1
        for i, e in enumerate(epochs):
            if e >= datetime.datetime.combine(START_DATE, datetime.time.min):
                idx_start_forecast = i
                break
        
        if idx_start_forecast == -1:
            # No data after start date in calib set? (Should be there since we merged start_sats)
            # This means START_DATE is before any available epoch, which shouldn't happen with merged data.
            # If it does, we can't calibrate.
            print(f"DEBUG: idx_start_forecast is -1 for {sid}. Start Date: {START_DATE}, First Epoch: {epochs[0]}")
            continue
            
        # Calibration range: 0 to idx_start_forecast
        # We want to fit the trajectory from t_start_calib (index 0) to t_start_forecast
        
        # Run calibration
        # Note: calibrate_alongtrack_dv_on_range takes start_index, end_index
        # It optimizes dv at start_index.
        try:
            dv_opt, rmse = main.calibrate_alongtrack_dv_on_range(
                sat, 0, idx_start_forecast, 
                forces=['central', 'J2', 'J3', 'J4', 'Sun', 'Moon', 'SRP'],
                verbose=False
            )
            # print(f"Calibrated {sid}: dv={dv_opt:.4f}, rmse={rmse:.2f}")
        except Exception:
            dv_opt = 0.0
            
        # Propagate from index 0 (Day -3) with dv_opt
        # We need to go up to Day 14
        # Total duration from Day -3 to Day 14 is CALIBRATION_DAYS + DURATION_DAYS
        
        epoch0 = epochs[0]
        x0, y0, z0, vx0, vy0, vz0 = main.sat_state_eci(sat, 0)
        v0 = (vx0, vy0, vz0)
        
        # Apply dv
        if dv_opt != 0.0:
            # Need helper from main, but it's private. Let's reimplement or import.
            # _apply_alongtrack_delta_v is not exported.
            # Re-implement simple version: v_new = v + dv * (v/|v|)
            v_mag = math.sqrt(vx0**2 + vy0**2 + vz0**2)
            if v_mag > 0:
                v0 = (vx0 * (1 + dv_opt/v_mag), vy0 * (1 + dv_opt/v_mag), vz0 * (1 + dv_opt/v_mag))
                
        r, v = (x0, y0, z0), v0
        current_epoch = epoch0
        
        # Target: Day 14 end
        end_time = datetime.datetime.combine(START_DATE, datetime.time.min) + datetime.timedelta(days=DURATION_DAYS)
        
        pred_list = []
        
        # Propagate loop
        # We step by 60 minutes
        step_out = 60
        step_sec = 60.0
        
        while current_epoch < end_time:
            next_epoch = current_epoch + datetime.timedelta(minutes=step_out)
            r, v = propagate_rk4(r, v, current_epoch, next_epoch, step=step_sec, forces=['central', 'J2', 'J3', 'J4', 'Sun', 'Moon', 'SRP'])
            current_epoch = next_epoch
            
            # Store all points for calibrated (including negative time relative to Start Day)
            pred_list.append((current_epoch, (*r, *v)))
                
        calibrated_predictions[sid] = pred_list

    # 4. Calculate Errors
    print("Calculating Errors...")
    
    def compute_rms_curve(predictions, label):
        time_points = []
        rms_errors = []
        if not predictions: return [], []
        
        sample_sid = next(iter(predictions))
        sample_preds = predictions[sample_sid]
        
        start_t = datetime.datetime.combine(START_DATE, datetime.time.min)
        
        # Align to sample timestamps
        for t, _ in sample_preds:
            errors = []
            for sid, pred_list in predictions.items():
                pred = next((state for pt, state in pred_list if pt == t), None)
                if not pred: continue
                
                if sid not in truth_sats: continue
                if sid.startswith('R') or sid.startswith('S'): continue
                
                truth_pos = get_truth_position(truth_sats[sid], t)
                if not truth_pos: continue
                
                px, py, pz = pred[:3]
                tx, ty, tz = truth_pos
                dist = math.sqrt((px-tx)**2 + (py-ty)**2 + (pz-tz)**2)
                
                if dist < 500000:
                    errors.append(dist)
            
            if errors:
                rms = np.sqrt(np.mean(np.array(errors)**2))
                dt_days = (t - start_t).total_seconds() / 86400.0
                time_points.append(dt_days)
                rms_errors.append(rms / 1000.0)
                
        return time_points, rms_errors

    t_uncal, err_uncal = compute_rms_curve(sky_image_uncal.predictions, "Uncalibrated")
    t_cal, err_cal = compute_rms_curve(calibrated_predictions, "Calibrated")
    
    print(f"Uncalibrated points: {len(t_uncal)}")
    if len(t_uncal) > 0:
        print(f"Uncalibrated sample: {t_uncal[:5]}, {err_uncal[:5]}")
        
    print(f"Calibrated points: {len(t_cal)}")
    if len(t_cal) > 0:
        # Print sample around t=0 (which might be in the middle now)
        # Find index where t >= 0
        idx_zero = next((i for i, t in enumerate(t_cal) if t >= 0), 0)
        print(f"Calibrated sample (around t=0): {t_cal[idx_zero:idx_zero+5]}, {err_cal[idx_zero:idx_zero+5]}")
            
    # 5. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t_uncal, err_uncal, label='Uncalibrated (Start Day 0)', linestyle='--')
    plt.plot(t_cal, err_cal, label='Calibrated (Start Day -3)', linewidth=2)
    plt.xlabel('Propagation Time (days from Day 0)')
    plt.ylabel('RMS Position Error (km)')
    plt.title(f'Propagation Accuracy: Calibrated vs Uncalibrated\n(Start: {START_DATE})')
    plt.grid(True)
    plt.legend()
    
    out_file = "static/accuracy_comparison_v2.png"
    plt.savefig(out_file)
    print(f"Comparison plot saved to {out_file}")

if __name__ == "__main__":
    run_analysis()
