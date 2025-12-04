
import os
import sys
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Add current directory to path so we can import main
sys.path.append(os.getcwd())

from main import ensure_latest_brdc, load_rinex_dir, test_propagation_errors, RINEX_DIR, SkyImage, Satellite

def run_test():
    print("Attempting to ensure latest BRDC file...")
    try:
        # Try to get data for the last 2 days
        files = ensure_latest_brdc(max_days_back=2)
        print(f"Got files: {files}")
    except Exception as e:
        print(f"Failed to download BRDC: {e}")
        print("Checking if we have any local files...")
        if not os.listdir(RINEX_DIR):
            print("No local files found. Cannot proceed with test.")
            return

    print("Loading RINEX data...")
    # Reload satellites from the directory
    raw_data = load_rinex_dir(RINEX_DIR)
    satellites_dict = {sid: Satellite(sid, entries) for sid, entries in raw_data.items()}
    # Update the global satellites dict if needed, though we are passing it explicitly
    
    print(f"Loaded {len(satellites_dict)} satellites.")
    
    # Run propagation test for GPS
    print("Running propagation test for GPS (G)...")
    try:
        summary = test_propagation_errors(
            satellites_dict, 
            constellation='G', 
            show_plot=False, # Don't show plot in non-interactive
            verbose=True
        )
        print("\nTest Summary:")
        print(summary)
        
        # Save summary to file
        summary.to_csv("propagation_error_summary.csv", index=False)
        print("Saved summary to propagation_error_summary.csv")
        
    except Exception as e:
        print(f"Error during propagation test: {e}")

if __name__ == "__main__":
    run_test()
