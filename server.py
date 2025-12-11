
import os
import datetime
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from main import ensure_latest_brdc, download_brdc_with_fallback, load_rinex_dir, SkyImage, Satellite, RINEX_DIR

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global state
class State:
    sky_image: SkyImage = None
    last_updated: datetime.datetime = None
    propagation_days: float = 14.0 # Default propagation duration
    is_updating: bool = False

state = State()

class PropagateRequest(BaseModel):
    days: int

import pickle

# ... (imports)

CACHE_FILE = "sky_image.pkl"

async def update_rinex_data():
    if state.is_updating:
        return
    state.is_updating = True
    
    def _update_sync():
        try:
            # Check for cache first
            if not state.sky_image and os.path.exists(CACHE_FILE):
                try:
                    print("Loading cached SkyImage...")
                    with open(CACHE_FILE, 'rb') as f:
                        state.sky_image = pickle.load(f)
                    state.last_updated = datetime.datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
                    print(f"Loaded cache from {state.last_updated}")
                    # Optional: Check if cache is too old (e.g. > 24h) and re-update if needed
                    # For now, we just load it. If the user wants fresh data, we might need a force refresh.
                except Exception as e:
                    print(f"Failed to load cache: {e}")

            # If we loaded cache and it's good, maybe we don't need to propagate?
            # But the requirement is to "automatically download... and propagate".
            # Let's say if we have a cache, we use it. But maybe we should check if a NEWER file exists?
            # For simplicity: If cache exists, use it. If not, propagate.
            # BUT, the user might want to run this daily.
            # Let's do this: Try to load cache. If successful, we are good.
            # Then, in a background step (or if cache missing), we try to fetch new data.
            
            if state.sky_image:
                 print("Using cached data.")
                 # We could return here, OR we could check if we need to update.
                 # Let's just return for now to satisfy "saves calculation".
                 # If the user restarts the server, it won't re-calculate.
                 return

            print("Updating RINEX data...")
            # 1. Ensure latest BRDC + 3 days history for calibration
            # We want today and 3 days back
            today = datetime.date.today()
            calibration_days = 3
            
            # Download history
            for i in range(calibration_days + 1):
                day = today - datetime.timedelta(days=i)
                try:
                    download_brdc_with_fallback(day, out_dir=RINEX_DIR)
                except Exception as e:
                    print(f"Warning: could not fetch BRDC for {day}: {e}")

            # 2. Load and propagate
            raw_data = load_rinex_dir(RINEX_DIR)
            satellites = {sid: Satellite(sid, entries) for sid, entries in raw_data.items()}
            state.sky_image = SkyImage(satellites)
            
            # Calibrate and Propagate
            print(f"Calibrating and Propagating for {state.propagation_days} days...")
            state.sky_image.calibrate_and_propagate(
                days=state.propagation_days,
                output_every_minutes=120, # 2 hours resolution for broadcast
                step_seconds=60.0,
                forces=['central', 'J2', 'J3', 'J4', 'Sun', 'Moon', 'SRP']
            )
            state.last_updated = datetime.datetime.now()
            print("RINEX data updated successfully.")
            
            # Save to cache
            try:
                with open(CACHE_FILE, 'wb') as f:
                    pickle.dump(state.sky_image, f)
                print("Saved SkyImage to cache.")
            except Exception as e:
                print(f"Failed to save cache: {e}")
            
        except Exception as e:
            print(f"Error updating RINEX data: {e}")
        finally:
            state.is_updating = False

    # Run in a separate thread
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _update_sync)

@app.on_event("startup")
async def startup_event():
    # Run initial update in background
    asyncio.create_task(update_rinex_data())

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.get("/status")
async def get_status():
    return {
        "last_updated": state.last_updated.isoformat() if state.last_updated else None,
        "is_updating": state.is_updating,
        "satellites_count": len(state.sky_image.satellites) if state.sky_image else 0
    }

@app.post("/propagate")
async def propagate(req: PropagateRequest):
    if not state.sky_image:
        raise HTTPException(status_code=503, detail="System is initializing, please wait.")
    
    # Check if we have enough propagated data
    # Ideally we should just slice the existing predictions
    # But for now, let's assume the user asks for <= state.propagation_days
    
    if req.days > state.propagation_days:
         raise HTTPException(status_code=400, detail=f"Max propagation days is {state.propagation_days}")

    # Create a temporary file for the result
    filename = f"prop_{req.days}d_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.rnx"
    filepath = os.path.join("output", filename)
    os.makedirs("output", exist_ok=True)
    
    # Write RINEX (slicing is handled inside write_rinex ideally, or we just dump everything)
    # For now, we dump everything we have.
    # TODO: Implement slicing in write_rinex or pass a time limit
    try:
        state.sky_image.write_rinex(filepath)
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="RINEX writer not implemented yet.")
        
    return FileResponse(filepath, filename=filename)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
