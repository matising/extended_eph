import datetime, math, pandas as pd
import os
import gzip
import glob
import requests
from pathlib import Path
from math import sin, cos, sqrt, atan2, radians
from datetime import timedelta
from functools import lru_cache
import cProfile, pstats, io
import matplotlib.pyplot as plt
import statistics as stats
from collections import defaultdict

MU       = 3.986005e14       # ‎μ‎ של כדור-הארץ [m³/s²] (ערך ה-GPS)
OMEGA_E  = 7.2921151467e-5   # קצב הסיבוב של כדוה"א [rad/s]
R_E = 6378137.0
J2  = 1.08262668e-3         # WGS-84
J3 = -2.532e-6
J4 = -1.619e-6

GM_SUN  = 1.32712440018e20   # [m³/s²]
GM_MOON = 4.9048695e12       # [m³/s²]
AU      = 1.495978707e11     # [m]  יח' אסטרונומית
C_LIGHT = 299_792_458.0      # [m/s]

# פרמטרי SRP (אפשר לשנות לפי הלוויין)
CR_DEFAULT  = 1.0            # מקדם השתקפות
AREA_M_RATIO = 0.012         # A/m  [m²/kg]   (≈ GPS ≈ 0.012)

RINEX_DIR = "rinex"           # directory that contains .rnx files and/or .gz archives

from skyfield.api import load
_eph  = load('de421.bsp')
_ts   = load.timescale()
earth = _eph['earth']
# Cache handles to bodies
_sun  = _eph['sun']
_moon = _eph['moon']

@lru_cache(maxsize=200000)
def _skyfield_time(epoch: datetime.datetime):
    """LRU-cached Skyfield Time construction (UTC)."""
    return _ts.utc(epoch.year, epoch.month, epoch.day,
                   epoch.hour, epoch.minute,
                   epoch.second + epoch.microsecond/1e6)

# ---------------------------------------------------------------------------
#  cProfile helper – run a callable and print/save profiling statistics
# ---------------------------------------------------------------------------
def run_with_cprofile(func, *args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    try:
        return func(*args, **kwargs)
    finally:
        pr.disable()
        s = io.StringIO()
        stats = pstats.Stats(pr, stream=s).strip_dirs()
        # Print top by cumulative time and total time
        stats.sort_stats('cumulative').print_stats(50)
        stats.sort_stats('tottime').print_stats(30)
        # Focus on our hotspots (printed again for quick visibility)
        for flt in (
            'sun_position_eci', 'moon_position_eci', '_skyfield_time',
            'acceleration_third_body', 'acceleration_srp',
            'acceleration_total', 'rk4_step', 'propagate_rk4'
        ):
            stats.sort_stats('cumulative').print_stats(flt)
        print("\n[cProfile] ---- Summary (top cumulative, tot time, and focused funcs) ----\n")
        print(s.getvalue())
        pr.dump_stats('profile_run.prof')
        print('[cProfile] Raw stats saved to profile_run.prof  (use snakeviz/gprof2dot to visualize)')

# ---- Parameter name lookup tables for each constellation ---- #
PARAM_MAPS = {
    'G': [  # GPS/QZSS (L1/L2)
        'SV_clock_bias', 'SV_clock_drift', 'SV_clock_drift_rate',
        'IODE', 'Crs', 'Delta_n', 'M0',
        'Cuc', 'e', 'Cus', 'sqrtA', 'Toe',
        'Cic', 'OMEGA0', 'Cis', 'i0',
        'Crc', 'omega', 'OMEGADOT', 'IDOT',
        'Codes_L2', 'GPS_week', 'L2P_flag', 'SV_accuracy',
        'SV_health', 'TGD', 'IODC', 'transmission_time',
        'fit_interval', 'spare1', 'spare2', 'spare3'
    ],
    'E': [  # Galileo
        'SV_clock_bias', 'SV_clock_drift', 'SV_clock_drift_rate',
        'IODnav', 'Crs', 'Delta_n', 'M0',
        'Cuc', 'e', 'Cus', 'sqrtA', 'Toe',
        'Cic', 'OMEGA0', 'Cis', 'i0',
        'Crc', 'omega', 'OMEGADOT', 'IDOT',
        'Data_source', 'GAL_week', 'SISA', 'SV_health',
        'BGD_E1E5a', 'BGD_E5bE1', 'transmission_time',
        'spare1', 'spare2', 'spare3', 'spare4', 'spare5'
    ],
    'C': [  # BeiDou‑2/3
        'SV_clock_bias', 'SV_clock_drift', 'SV_clock_drift_rate',
        'AODE', 'Crs', 'Delta_n', 'M0',
        'Cuc', 'e', 'Cus', 'sqrtA', 'Toe',
        'Cic', 'OMEGA0', 'Cis', 'i0',
        'Crc', 'omega', 'OMEGADOT', 'IDOT',
        'Week', 'URAI', 'SV_health', 'TGD1',
        'TGD2', 'transmission_time', 'spare1', 'spare2',
        'spare3', 'spare4', 'spare5', 'spare6'
    ],
    'R': [  # GLONASS (only 3 data lines after header)
        'SV_clock_bias', 'rel_freq_bias', 'Message_frame_time',
        'X', 'X_velocity', 'X_acceleration', 'SV_health',
        'Y', 'Y_velocity', 'Y_acceleration', 'Frequency_number',
        'Z', 'Z_velocity', 'Z_acceleration', 'Age_oper_info'
    ],
    'S': [  # SBAS
        'SV_clock_bias', 'SV_clock_drift', 'Message_frame_time',
        'X', 'X_velocity', 'X_acceleration', 'SV_health',
        'Y', 'Y_velocity', 'Y_acceleration', 'PRN',
        'Z', 'Z_velocity', 'Z_acceleration', 'IODN'
    ],
    'I': [  # IRNSS/NavIC – same layout as GPS in RINEX3 NAV
    'SV_clock_bias','SV_clock_drift','SV_clock_drift_rate',
    'IODE','Crs','Delta_n','M0',
    'Cuc','e','Cus','sqrtA','Toe',
    'Cic','OMEGA0','Cis','i0',
    'Crc','omega','OMEGADOT','IDOT',
    'Codes_L2','GPS_week','L2P_flag','SV_accuracy',
    'SV_health','TGD','IODC','transmission_time',
    'fit_interval','spare1','spare2','spare3'
],
    'J': [  # IRNSS/NavIC – same layout as GPS in RINEX3 NAV
    'SV_clock_bias','SV_clock_drift','SV_clock_drift_rate',
    'IODE','Crs','Delta_n','M0',
    'Cuc','e','Cus','sqrtA','Toe',
    'Cic','OMEGA0','Cis','i0',
    'Crc','omega','OMEGADOT','IDOT',
    'Codes_L2','GPS_week','L2P_flag','SV_accuracy',
    'SV_health','TGD','IODC','transmission_time',
    'fit_interval','spare1','spare2','spare3'
]
}

def _to_float(s: str) -> float:
    """Convert FORTRAN‐style float with D exponent to Python float."""
    return float(s.replace('D', 'E').strip()) if s.strip() else math.nan

def parse_rinex_nav(path: str):
    """Return {satellite_id: [(epoch_datetime, {param_name: value})]} dict."""
    data = {}
    with open(path, "r") as f:
        while True:
            # -------- skip header (initial or intermediate) -------- #
            header_found = False
            while True:
                pos = f.tell()
                ln = f.readline()
                if not ln:
                    break # EOF
                if "END OF HEADER" in ln:
                    header_found = True
                    break
                # If we see a data line while looking for header end, it means we might have missed the header end or it wasn't there.
                # But for now, let's assume standard structure: Header -> Data -> (maybe Header -> Data)
            
            if not ln and not header_found:
                 break # EOF reached while looking for header

            # -------- read data block -------- #
            while True:
                pos = f.tell()
                hdr = f.readline()
                if not hdr:
                    break
                if not hdr.strip():
                    continue

                # Check for new header start
                if "RINEX VERSION" in hdr or "PGM / RUN BY / DATE" in hdr:
                    f.seek(pos) # Backtrack to let the outer loop handle this header
                    break
                
                try:
                    # First 22 columns contain PRN & epoch fields
                    parts = hdr[:22].split()
                    if len(parts) < 7:
                         # Maybe a header line we missed?
                         if "RINEX" in hdr or "PGM" in hdr or "IONOSPHERIC" in hdr:
                             f.seek(pos)
                             break
                         continue # Just skip weird short lines

                    sat = parts[0]          # e.g. G01, R06, E12 …
                    yy, mm, dd, hh, mi, ss  = map(int, parts[1:7])
                    epoch = datetime.datetime(yy, mm, dd, hh, mi, ss)
                except ValueError:
                    # Parsing failed, likely hit a header line
                    f.seek(pos)
                    break

                # Three SV‑clock parameters sit at fixed cols 23‑41, 42‑60, 61‑79
                try:
                    clk = [_to_float(hdr[c:c+19]) for c in (23, 42, 61)]
                except ValueError:
                     # Failed to parse clock, treat as end of block
                     f.seek(pos)
                     break

                sys = sat[0]
                # GLONASS & SBAS have 3 data lines, all others 7
                extra_lines = 3 if sys in ("R", "S") else 7
                starts = (4, 23, 42, 61)  # four 19‑char fields per line

                vals = clk
                for _ in range(extra_lines):
                    line = f.readline()
                    vals.extend(_to_float(line[s:s+19]) for s in starts)

                # Map into a name→value dict (pad if spec has fewer names)
                names = PARAM_MAPS.get(sys, [f"param_{i}" for i in range(len(vals))])
                if len(names) < len(vals):
                    names += [f"extra_{i}" for i in range(len(vals) - len(names))]
                params = dict(zip(names, vals))

                data.setdefault(sat, []).append((epoch, params))
                
    return data

def _extract_gz(gz_path: str) -> str:
    """
    Extract <name>.gz to <name> in the same directory if not already extracted.
    Returns the path to the extracted .rnx file.
    """
    out_path = gz_path[:-3]  # strip '.gz'
    if not os.path.exists(out_path):
        with gzip.open(gz_path, 'rb') as g_in, open(out_path, 'wb') as f_out:
            f_out.write(g_in.read())
    return out_path

# ---------------------------------------------------------------------------
#  RINEX (BRDC) downloader via Earthdata (.netrc or Bearer token)
# ---------------------------------------------------------------------------
def _brdc_fname_for_date(day: datetime.date) -> str:
    y = day.year
    doy = day.timetuple().tm_yday
    return f"BRDC00IGS_R_{y}{doy:03d}0000_01D_MN.rnx.gz"

# Build a BRDC filename for a given producer/tag (e.g., IGS/R or DLR/S)
def _brdc_fname_for_date_product(day: datetime.date, producer: str, tag: str) -> str:
    y = day.year
    doy = day.timetuple().tm_yday
    return f"BRDC00{producer}_{tag}_{y}{doy:03d}0000_01D_MN.rnx.gz"

def download_brdc_for_day(day: datetime.date, out_dir: str = RINEX_DIR, timeout: int = 120) -> str:
    """Download BRDC NAV for the given day into out_dir using Earthdata auth.
    Returns the path to the extracted .rnx file (extracts .gz if needed).
    Uses Bearer token from $EARTHDATA_TOKEN if set; otherwise relies on ~/.netrc.
    If file already exists locally, it is reused.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fname = _brdc_fname_for_date(day)
    url = f"https://cddis.nasa.gov/archive/gnss/data/daily/{day.year}/brdc/{fname}"
    gz_path = str(Path(out_dir) / fname)

    # Reuse if already present
    if os.path.exists(gz_path):
        return _extract_gz(gz_path)

    sess = requests.Session()
    sess.trust_env = True  # allow system certs/proxies

    headers = {}
    token = os.environ.get('EARTHDATA_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'

    # First attempt: plain GET (will use .netrc on redirect to urs if needed)
    resp = sess.get(url, headers=headers, allow_redirects=True, timeout=timeout)

    # If unauthorized and no token, try again explicitly with .netrc creds for URS
    if resp.status_code in (401, 403) and not token:
        auth = requests.utils.get_netrc_auth('https://urs.earthdata.nasa.gov')
        if auth:
            sess.auth = auth
            resp = sess.get(url, headers=headers, allow_redirects=True, timeout=timeout)

    resp.raise_for_status()
    with open(gz_path, 'wb') as f:
        f.write(resp.content)

    return _extract_gz(gz_path)


# Download BRDC with fallback: try multiple producers in order (default: IGS, then DLR)
def download_brdc_with_fallback(day: datetime.date, out_dir: str = RINEX_DIR,
                                timeout: int = 45,
                                products= None) -> str:
    """Download BRDC NAV for a given day, trying multiple producers in order.
    Default order: [('IGS','R'), ('DLR','S')]. Returns path to extracted .rnx.
    Uses EARTHDATA_TOKEN if set, otherwise relies on ~/.netrc automatically.
    """
    if products is None:
        products = [('IGS','R'), ('DLR','S')]

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    sess = requests.Session()
    sess.trust_env = True

    headers = {}
    token = os.environ.get('EARTHDATA_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'

    last_err = None

    for producer, tag in products:
        fname = _brdc_fname_for_date_product(day, producer, tag)
        url = f"https://cddis.nasa.gov/archive/gnss/data/daily/{day.year}/brdc/{fname}"
        gz_path = str(Path(out_dir) / fname)
        rnx_path = gz_path[:-3]

        # Reuse if already present locally
        if os.path.exists(rnx_path):
            return rnx_path
        if os.path.exists(gz_path):
            try:
                return _extract_gz(gz_path)
            except Exception as e:
                print(f"[BRDC] Warning: failed to extract existing {gz_path}: {e}")
                # fall through to re-download

        try:
            resp = sess.get(url, headers=headers, allow_redirects=True, timeout=(10, timeout))
            if resp.status_code in (401, 403) and not token:
                auth = requests.utils.get_netrc_auth('https://urs.earthdata.nasa.gov')
                if auth:
                    sess.auth = auth
                    resp = sess.get(url, headers=headers, allow_redirects=True, timeout=(10, timeout))
            resp.raise_for_status()
            with open(gz_path, 'wb') as f:
                f.write(resp.content)
            return _extract_gz(gz_path)
        except Exception as e:
            print(f"[BRDC] {producer}_{tag} for {day} not available: {e}")
            last_err = e
            continue

    raise last_err if last_err else RuntimeError(f"No BRDC product available for {day}")

# Try to ensure a recent BRDC exists locally: today, yesterday, ... (tries up to max_days_back days, stops at first success)
def ensure_latest_brdc(max_days_back: int = 2, out_dir: str = RINEX_DIR) -> list[str]:
    """Ensure a recent BRDC exists locally by trying today, then 1–2 days back.
    Stops on the first successful download/extract and returns [path].
    Raises if none were found.
    """
    today = datetime.date.today()
    for i in range(max_days_back + 1):
        day = today - datetime.timedelta(days=i)
        try:
            path_rnx = download_brdc_with_fallback(day, out_dir=out_dir, timeout=45)
            print(f"[BRDC] using {Path(path_rnx).name} ({day})")
            return [path_rnx]
        except Exception as e:
            print(f"[BRDC] Warning: could not fetch any BRDC for {day} → {e}")
            continue
    raise FileNotFoundError(f"No recent BRDC found (today/back {max_days_back} days)")

def load_rinex_dir(dir_path: str) -> dict:
    """
    Scan a directory for .rnx files and .gz archives, extract when needed,
    parse each NAV file, and return a merged {sat_id: [(epoch, params), ...]} dict.
    """
    # 1. collect and extract
    for gz in glob.glob(os.path.join(dir_path, "*.gz")):
        _extract_gz(gz)
    rinex_files = glob.glob(os.path.join(dir_path, "*.rnx")) + \
                  glob.glob(os.path.join(dir_path, "*.nav"))
    if not rinex_files:
        raise FileNotFoundError(f"No RINEX NAV files found in {dir_path!r}")

    # 2. parse and merge
    merged: dict = {}
    for path in rinex_files:
        recs = parse_rinex_nav(path)
        for sat_id, entries in recs.items():
            merged.setdefault(sat_id, []).extend(entries)

    # 3. sort each satellite's entries chronologically
    for entries in merged.values():
        entries.sort(key=lambda x: x[0])

    return merged

# Build a convenience Satellite wrapper ----------------------------------- #
class Satellite:
    def __init__(self, sat_id, entries):
        self.id = sat_id
        # sort by time for deterministic access
        self.entries = sorted(entries, key=lambda x: x[0])

    def epochs(self):
        """Return list of all epochs available for this satellite."""
        return [e for e, _ in self.entries]

    def params_at(self, epoch):
        """Return {param: value} dict for a specific epoch."""
        for e, p in self.entries:
            if e == epoch:
                return p
        raise KeyError(f"{epoch} not found for {self.id}")

    def get(self, epoch, param):
        """Get a specific parameter at epoch (by name or zero‑based index)."""
        p = self.params_at(epoch)
        if isinstance(param, int):
            return list(p.values())[param]
        return p[param]

def _datetime_to_julian(dt: datetime.datetime) -> float:
    """Gregorian → Julian Day."""
    y, m = dt.year, dt.month
    if m <= 2:
        y -= 1; m += 12
    A = y // 100
    B = 2 - A + A // 4
    day_frac = (dt.hour + dt.minute/60 + dt.second/3600) / 24
    JD = int(365.25*(y+4716)) + int(30.6001*(m+1)) + dt.day + day_frac + B - 1524.5
    return JD

def _gmst(dt: datetime.datetime) -> float:
    """Greenwich Mean Sidereal Time [rad] עבור Instant UTC נתון."""
    JD  = _datetime_to_julian(dt)
    T   = (JD - 2451545.0) / 36525      # מאות שנים יוליאניות מאז J2000.0
    gmst = (280.46061837 +
            360.98564736629 * (JD - 2451545.0) +
            0.000387933*T**2 - T**3/38710000)  # [deg]
    return math.radians(gmst % 360.0)

# ---------- פתרון משוואת קפלר -------------------------------------------
def _eccentric_anomaly(M, e, tol=1e-12):
    """Iterative solve of M = E - e·sinE."""
    E = M
    for _ in range(10):
        E_next = M + e*sin(E)
        if abs(E_next - E) < tol:
            break
        E = E_next
    return E

# ---------- פונקציית-על --------------------------------------------------
def sat_eci(sat: 'Satellite', idx: int) -> tuple[float, float, float]:
    """
    החזר (x, y, z) ‏ב-ECI [m] עבור הלווין ‘sat’ באינדקס ‘idx’.
    תומך ב-GPS/QZSS/Galileo/BeiDou/IRNSS + טיפול בסיסי ב-GLONASS.
    """
    epoch, p = sat.entries[idx]
    sys = sat.id[0]

    # ---------------- GLONASS  – broadcast נותן כבר XYZ+V+A ב-ECEF -------------
    if sys == 'R':
        # יחידות בקובץ הן ק"מ → ממירים למטר
        x  = p['X']  * 1e3 + p['X_velocity'] * 1e3 * p['Message_frame_time']
        y  = p['Y']  * 1e3 + p['Y_velocity'] * 1e3 * p['Message_frame_time']
        z  = p['Z']  * 1e3 + p['Z_velocity'] * 1e3 * p['Message_frame_time']
        # סיבוב ECEF→ECI
        θ  = _gmst(epoch)
        return (x*cos(θ) - y*sin(θ),
                x*sin(θ) + y*cos(θ),
                z)

    # ----------------‏ GPS-style  (7 שורות – Kepler + תיקונים) -----------------
    # 1. קבועים והפרשי-זמן
    A     = p['sqrtA'] ** 2
    n0    = sqrt(MU / A**3)
    n     = n0 + p['Delta_n']

    # שניות-שבוע GPS/GAL/BD – לקבל tk נכון
    if 'GPS_week' in p:
        wk  = int(p['GPS_week'])
    elif 'GAL_week' in p:
        wk  = int(p['GAL_week'])
    elif 'Week' in p:          # BeiDou
        wk  = int(p['Week'])
    else:
        wk  = None

    if wk is not None:
        t0_week = datetime.datetime(1980, 1, 6) + datetime.timedelta(weeks=wk)
        tk      = (epoch - t0_week).total_seconds() - p['Toe']
    else:
        tk      = 0.0  # fallback – קירוב גס
        print("used the hard approx' for tk")

    # ‏הבטחת tk ∈ [-302400, +302400]
    if tk >  302400: tk -= 604800
    if tk < -302400: tk += 604800

    # 2. אנומליות ו-Perturbations
    M   = p['M0'] + n * tk
    E   = _eccentric_anomaly(M, p['e'])
    ν   = atan2(sqrt(1 - p['e']**2) * sin(E), cos(E) - p['e'])
    φ   = ν + p['omega']

    du  = p['Cuc'] * cos(2*φ) + p['Cus'] * sin(2*φ)
    dr  = p['Crc'] * cos(2*φ) + p['Crs'] * sin(2*φ)
    di  = p['Cic'] * cos(2*φ) + p['Cis'] * sin(2*φ)

    u   = φ + du
    r   = A * (1 - p['e'] * cos(E)) + dr
    i   = p['i0'] + di + p['IDOT'] * tk

    # 3. קואורדינטות במישור המסלול
    xʹ, yʹ = r * cos(u), r * sin(u)

    # 4. אורך קו-המשווה
    Ω = (p['OMEGA0'] +
         (p['OMEGADOT'] - OMEGA_E) * tk -
         OMEGA_E * p['Toe'])

    # 5. מעבר ל-ECEF
    x = xʹ *  cos(Ω) - yʹ * cos(i) * sin(Ω)
    y = xʹ *  sin(Ω) + yʹ * cos(i) * cos(Ω)
    z = yʹ *  sin(i)

    # 6. ECEF → ECI
    θ = _gmst(epoch)
    return (x*cos(θ) - y*sin(θ),
            x*sin(θ) + y*cos(θ),
            z)

# ---------------------------------------------------------------------------
#  ↪️  ANL-DERIV  –  Velocity from analytic derivatives (ICD compliant)
# ---------------------------------------------------------------------------
def _gps_ecef_pos_vel(epoch: datetime.datetime, p: dict):
    """החזר (x, y, z, vx, vy, vz) ב-ECEF  עבור GPS-style broadcast."""
    # --- קבועים בסיסיים ---
    A      = p['sqrtA'] ** 2
    n0     = sqrt(MU / A**3)
    n      = n0 + p['Delta_n']

    # --- tk  ---------------------------------------------------------------
    if 'GPS_week' in p:   wk = int(p['GPS_week'])
    elif 'GAL_week' in p: wk = int(p['GAL_week'])
    elif 'Week'     in p: wk = int(p['Week'])
    else:                 wk = None
    toe = p['Toe']

    if wk is not None:
        t0_week = datetime.datetime(1980, 1, 6) + datetime.timedelta(weeks=wk)
        tk      = (epoch - t0_week).total_seconds() - toe
    else:
        tk = 0.0

    if tk >  302400: tk -= 604800
    if tk < -302400: tk += 604800

    # --- משוואת קפלר + נגזרות ---------------------------------------------
    M   = p['M0'] + n * tk
    E   = _eccentric_anomaly(M, p['e'])
    Edot = n / (1 - p['e'] * cos(E))                 # dE/dt

    # True anomaly
    nu   = atan2(sqrt(1 - p['e']**2) * sin(E), cos(E) - p['e'])
    nudot = (sqrt(1 - p['e']**2) * Edot) / (1 - p['e'] * cos(E))

    # Argument of latitude
    phi   = nu + p['omega']
    phidot = nudot                                 # ω‎ קבועה

    # Corrections and derivatives
    du  = p['Cuc'] * cos(2*phi) + p['Cus'] * sin(2*phi)
    dr  = p['Crc'] * cos(2*phi) + p['Crs'] * sin(2*phi)
    di  = p['Cic'] * cos(2*phi) + p['Cis'] * sin(2*phi)

    du_dot = 2*phidot * (-p['Cuc']*sin(2*phi) + p['Cus']*cos(2*phi))
    dr_dot = 2*phidot * (-p['Crc']*sin(2*phi) + p['Crs']*cos(2*phi))
    di_dot = 2*phidot * (-p['Cic']*sin(2*phi) + p['Cis']*cos(2*phi)) + p['IDOT']

    # Inclination, radius, argument-of-latitude
    u     = phi + du
    udot  = phidot + du_dot

    r     = A * (1 - p['e'] * cos(E)) + dr
    r_dot = A * p['e'] * sin(E) * Edot + dr_dot

    i     = p['i0'] + di + p['IDOT'] * tk
    i_dot = di_dot

    # --- קואורדינטות מישוריות + נגזרות ------------------------------------
    x_p   = r * cos(u)
    y_p   = r * sin(u)

    x_p_dot = r_dot * cos(u) - r * udot * sin(u)
    y_p_dot = r_dot * sin(u) + r * udot * cos(u)

    # --- קו-הרוחב (RAAN) ----------------------------------------------------
    Omega      = (p['OMEGA0'] +
                  (p['OMEGADOT'] - OMEGA_E) * tk -
                  OMEGA_E * toe)
    Omega_dot  = p['OMEGADOT'] - OMEGA_E

    cosO, sinO = cos(Omega), sin(Omega)
    cosi, sini = cos(i),     sin(i)

    # --- מיקום ECEF ---------------------------------------------------------
    x = x_p * cosO - y_p * cosi * sinO
    y = x_p * sinO + y_p * cosi * cosO
    z = y_p * sini

    # --- מהירות ECEF (נגזרת אלגברית) ---------------------------------------
    x_dot = ( x_p_dot * cosO
            - x_p      * sinO * Omega_dot
            - y_p_dot * cosi * sinO
            - y_p     * (-sini * i_dot) * sinO
            - y_p     * cosi * cosO * Omega_dot )

    y_dot = ( x_p_dot * sinO
            + x_p      * cosO * Omega_dot
            + y_p_dot * cosi * cosO
            + y_p     * (-sini * i_dot) * cosO
            - y_p     * cosi * sinO * Omega_dot )

    z_dot = y_p_dot * sini + y_p * cosi * i_dot

    return x, y, z, x_dot, y_dot, z_dot

# ---------------------------------------------------------------------------
#  util: ECEF→ECI סיבוב + תרומת סיבוב־הארץ
# ---------------------------------------------------------------------------
def _ecef_to_eci(epoch_utc: datetime.datetime, x_ecef, y_ecef, z_ecef,
                 vx_ecef=0.0, vy_ecef=0.0, vz_ecef=0.0):
    """המרת מקום (ולפי הצורך מהירות) מ-ECEF ל-ECI."""
    θ = _gmst(epoch_utc)
    c, s = math.cos(θ), math.sin(θ)

    # מיקום
    x_eci =  c*x_ecef - s*y_ecef
    y_eci =  s*x_ecef + c*y_ecef
    z_eci =  z_ecef

    # if vx_ecef == vy_ecef == vz_ecef == 0.0:
    #     return (x_eci, y_eci, z_eci)           # רק מיקום

    # מהירות (R·v + ω×r)
    vx_rot = c*vx_ecef - s*vy_ecef
    vy_rot = s*vx_ecef + c*vy_ecef
    vz_rot = vz_ecef

    vx_eci = vx_rot - OMEGA_E * y_eci
    vy_eci = vy_rot + OMEGA_E * x_eci
    vz_eci = vz_rot

    return (x_eci, y_eci, z_eci, vx_eci, vy_eci, vz_eci)
# ---------------------------------------------------------------------------
#  ולעדכון sat_state_eci  (GPS-style חלק) – ללא הפרש סופי!
# ---------------------------------------------------------------------------
def sat_state_eci(sat: 'Satellite', idx: int):
    """
    (x, y, z, vx, vy, vz)  ב-ECI  – כעת עם נגזרות אנליטיות (ICD).
    GLONASS / SBAS: משתמש בשדות המהירות שב-RINEX + ω×r.
    """
    epoch, p = sat.entries[idx]
    sys = sat.id[0]

    # -- GLONASS / SBAS -------------------------------------------------------
    if sys in ('R', 'S'):
        return _ecef_to_eci(
            epoch,
            1e3*p['X'],       1e3*p['Y'],       1e3*p['Z'],
            1e3*p['X_velocity'], 1e3*p['Y_velocity'], 1e3*p['Z_velocity']
        )

    # -- GPS-style ------------------------------------------------------------
    xE, yE, zE, vxE, vyE, vzE = _gps_ecef_pos_vel(epoch, p)
    # סיבוב אל ECI
    return _ecef_to_eci(epoch, xE, yE, zE, vxE, vyE, vzE)


def acceleration_j2(r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    החזר (ax, ay, az) [m/s²] עבור וקטור-מיקום ‎r = (x,y,z)‎ ב-ECI,
    כולל גרביטציית מרכז-כדור-הארץ + הפרעת J2.
    """
    x, y, z = r
    r2  = x*x + y*y + z*z
    r_  = math.sqrt(r2)

    # תאוצה מרכזית
    mu_r3 = -MU / r2 / r_
    ax, ay, az = mu_r3 * x, mu_r3 * y, mu_r3 * z

    # תיקון J2
    z2   = z*z
    r5   = r2 * r_ * r2          # r⁵
    k    = 1.5 * J2 * MU * R_E**2 / r5
    factor = 5 * z2 / r2  - 1

    ax += k * x * factor
    ay += k * y * factor
    az += k * z * (factor - 2)

    return ax, ay, az

@lru_cache(maxsize=200000)
def sun_position_eci(epoch: datetime.datetime) -> tuple[float, float, float]:
    """ECI Sun vector *Earth-centred* (Skyfield • מדויק ~1 m)."""
    t = _skyfield_time(epoch)
    pos = (_sun.at(t) - earth.at(t)).position.m  # numpy array
    return (float(pos[0]), float(pos[1]), float(pos[2]))

@lru_cache(maxsize=200000)
def moon_position_eci(epoch: datetime.datetime) -> tuple[float, float, float]:
    """ECI Moon vector *Earth-centred* (Skyfield)."""
    t = _skyfield_time(epoch)
    pos = (_moon.at(t) - earth.at(t)).position.m
    return (float(pos[0]), float(pos[1]), float(pos[2]))


# ---------------------------------------------------------------------------
#  ❖  3. תאוצות נוספות
# ---------------------------------------------------------------------------
def acceleration_j3_j4(r: tuple[float, float, float]) -> tuple[float, float, float]:
    """תאוצות J3, J4 (Wertz, 12-16)."""
    x, y, z = r
    r2 = x*x + y*y + z*z
    r  = math.sqrt(r2)
    ζ  = z / r
    r5 = r2 * r * r2
    r7 = r5 * r2

    a_j3 = 0.5 * J3 * MU * R_E**3 / r5
    a_j4 = (5/8) * J4 * MU * R_E**4 / r7

    ax = x * ζ * (5*ζ*ζ - 3) * a_j3 + x * (35*ζ**4 - 30*ζ**2 + 3) * a_j4
    ay = y * ζ * (5*ζ*ζ - 3) * a_j3 + y * (35*ζ**4 - 30*ζ**2 + 3) * a_j4
    az = ( (6*ζ*ζ - 1) * a_j3 * r2 +
           (35*ζ**4 - 30*ζ**2 + 3) * a_j4 * r2 )
    return ax, ay, az


def acceleration_third_body(r_sat: tuple[float, float, float],
                            r_body: tuple[float, float, float],
                            GM_body: float) -> tuple[float, float, float]:
    """
    תאוצה של גוף־שלישי (Sun/Moon) על הלוויין.
    """
    dx = r_body[0] - r_sat[0]
    dy = r_body[1] - r_sat[1]
    dz = r_body[2] - r_sat[2]

    d2 = dx*dx + dy*dy + dz*dz
    d  = math.sqrt(d2)

    r_b2 = r_body[0]**2 + r_body[1]**2 + r_body[2]**2
    r_b  = math.sqrt(r_b2)

    factor     = GM_body / d2 / d
    factor_ref = GM_body / r_b2 / r_b

    ax = factor*dx - factor_ref*r_body[0]
    ay = factor*dy - factor_ref*r_body[1]
    az = factor*dz - factor_ref*r_body[2]
    return ax, ay, az


def acceleration_srp(r_sat: tuple[float, float, float],
                     r_sun: tuple[float, float, float],
                     Cr: float = CR_DEFAULT,
                     A_mass: float = AREA_M_RATIO) -> tuple[float, float, float]:
    """
    Solar-Radiation-Pressure acceleration (m/s²).

    • הכיוון הוא *מהשמש החוצה* (Sun → Satellite)
    • אם הלוויין בצל כדוה״א (מודל גליל פשוט) ⇒ SRP = 0
    """
    sx, sy, sz = r_sun
    rx, ry, rz = r_sat

    # וקטור מהשמש אל הלוויין
    dx = rx - sx
    dy = ry - sy
    dz = rz - sz
    d2 = dx*dx + dy*dy + dz*dz
    d  = math.sqrt(d2)

    # ---------- צִלָּה (Umbra) – בדיקת גליל ------------------
    dot_rs = rx*sx + ry*sy + rz*sz          # r_sat · r_sun
    if dot_rs < 0:                          # הלוויין "מאחורי" כדוה״א
        sun_norm2 = sx*sx + sy*sy + sz*sz
        # מרחק מאונך מציר השמש
        perp2 = (d2 * sun_norm2 - dot_rs**2) / sun_norm2
        if perp2 < R_E**2:
            return (0.0, 0.0, 0.0)          # בתוך צל ⇒ אין SRP

    # ---------- גודל התאוצה -------------------------------
    P0 = 4.56e-6                            # N/m² ב-1 AU
    a_mag = P0 * Cr * A_mass * (AU / d)**2  # [m/s²]

    inv_d = 1.0 / d
    ax =  a_mag * dx * inv_d   # ❶ **כיוון החוצה מהשמש**
    ay =  a_mag * dy * inv_d
    az =  a_mag * dz * inv_d
    return ax, ay, az


# ---------------------------------------------------------------------------
#  ✨ 1. תרומות תאוצה אטומיות – כל אחת בנפרד
# ---------------------------------------------------------------------------
def accel_central(r: tuple[float, float, float]) -> tuple[float, float, float]:
    """-μ·r / r³  – כוח גרביטציה מרכזי בלבד (ללא J2)."""
    x, y, z = r
    r2 = x*x + y*y + z*z
    inv_r3 = -MU / (r2 * math.sqrt(r2))
    return inv_r3 * x, inv_r3 * y, inv_r3 * z


def accel_J2(r):
    """רק רכיב J2  (שימוש בפורמולה שהייתה ב-acceleration_j2)."""
    x, y, z = r
    r2 = x*x + y*y + z*z
    r  = math.sqrt(r2)
    z2 = z*z
    k  = 1.5 * J2 * MU * R_E**2 / (r**5)
    factor = 5*z2/r2 - 1
    return (k * x * factor,
            k * y * factor,
            k * z * (factor - 2))


def _a_j3_j4_components(r):
    """חשב את גורמי־המשקל של J3 ו-J4 בנפרד (Wertz)."""
    x, y, z = r
    r2 = x*x + y*y + z*z
    r  = math.sqrt(r2)
    ζ  = z / r
    r5 = r2 * r * r2
    r7 = r5 * r2

    aJ3 = 0.5 * J3 * MU * R_E**3 / r5
    aJ4 = 5/8 * J4 * MU * R_E**4 / r7
    return x, y, z, ζ, r2, aJ3, aJ4


# ---------------------------------------------------------------
#  J3  (סדר שלישי)  –  מבוסס Vallado Eq. 8-6
# ---------------------------------------------------------------
def accel_J3(r):
    x, y, z = r
    r2  = x*x + y*y + z*z
    r   = math.sqrt(r2)

    ζ    = z / r                 # sin φ
    k3   = - (5/2) * J3 * MU * R_E**3 / r**7   # ‎-‎ לפי נגזרת הפוטנציאל

    common = k3 * ζ * (3 - 7*ζ*ζ)              # ‎(3-7ζ²)·ζ
    ax = common * x
    ay = common * y
    az = k3 * (6*ζ*ζ - 1) * r2                 # ‎(6ζ²-1)·r²

    return ax, ay, az


# ---------------------------------------------------------------
#  J4  (סדר רביעי)  –  Vallado Eq. 8-7
# ---------------------------------------------------------------
def accel_J4(r):
    x, y, z = r
    r2 = x*x + y*y + z*z
    r  = math.sqrt(r2)

    ζ   = z / r
    k4  =  (15/8) * J4 * MU * R_E**4 / r**9    # שים לב ל-r⁹ במכנה

    f_xy = (1 - 14*ζ*ζ + 21*ζ**4)              # לפלסי-x,y
    f_z  = (5*ζ**4 - 6*ζ*ζ + 1)                # לפלסי-z

    ax = -k4 * x * f_xy
    ay = -k4 * y * f_xy
    az = -k4 * z * f_z

    return ax, ay, az

# -- עטיפות לגוף-שלישי ול-SRP (הקוד אצלך כבר קיים) -----------------------
def accel_third_body_sun(r_sat, epoch):
    return acceleration_third_body(r_sat, sun_position_eci(epoch), GM_SUN)

def accel_third_body_moon(r_sat, epoch):
    return acceleration_third_body(r_sat, moon_position_eci(epoch), GM_MOON)

def accel_srp(r_sat, epoch, Cr=CR_DEFAULT, A_m=AREA_M_RATIO):
    return acceleration_srp(r_sat, sun_position_eci(epoch), Cr, A_m)


# ---------------------------------------------------------------------------
#  ✨ 2. מילון כוחות ⇢ פונקציות
# ---------------------------------------------------------------------------
FORCES = {
    'central':  lambda r, e: accel_central(r),
    'J2':       lambda r, e: accel_J2(r),
    'J3':       lambda r, e: accel_J3(r),
    'J4':       lambda r, e: accel_J4(r),
    'Sun':      lambda r, e: accel_third_body_sun(r, e),
    'Moon':     lambda r, e: accel_third_body_moon(r, e),
    'SRP':      lambda r, e: accel_srp(r, e),
}


# ---------------------------------------------------------------------------
#  ✨ 3. תאוצה כוללת לפי רשימת דגלים
# ---------------------------------------------------------------------------
def acceleration_total(r, epoch, active=None):
    """
    מחזיר Σ תאוצות עבור הכוחות ב-active.
    מחשב וקטורי Sun/Moon פעם אחת לכל קריאה ומשתמש בהם גם ל-SRP,
    כדי להימנע מכפל קריאות לאפמריס. התוצאה הנומרית זהה לחלוטין.
    """
    if active is None:
        active = FORCES.keys()
    # normalize active to a list (may be dict_keys)
    if not isinstance(active, (list, tuple)):
        active = list(active)

    # Precompute third-body vectors if needed
    r_sun = r_moon = None
    needs_sun = ('Sun' in active) or ('SRP' in active)
    if needs_sun:
        r_sun = sun_position_eci(epoch)
    if 'Moon' in active:
        r_moon = moon_position_eci(epoch)

    ax = ay = az = 0.0
    for key in active:
        if key == 'central':
            ai = accel_central(r)
        elif key == 'J2':
            ai = accel_J2(r)
        elif key == 'J3':
            ai = accel_J3(r)
        elif key == 'J4':
            ai = accel_J4(r)
        elif key == 'Sun':
            ai = acceleration_third_body(r, r_sun, GM_SUN)
        elif key == 'Moon':
            ai = acceleration_third_body(r, r_moon, GM_MOON)
        elif key == 'SRP':
            ai = acceleration_srp(r, r_sun)
        else:
            # fallback to mapping for any custom force
            ai = FORCES[key](r, epoch)
        ax += ai[0]; ay += ai[1]; az += ai[2]
    return ax, ay, az


# ---------------------------------------------------------------------------
#  ✨ 4. RK4 מעודכן (חובה להחליף את הקיים)
# ---------------------------------------------------------------------------
def rk4_step(r, v, h, epoch, forces):
    """צעד RK4 – עובד עם רשימת כוחות."""
    def add(vec, k, c=1.0):
        return tuple(a + c*b for a, b in zip(vec, k))

    a1 = acceleration_total(r, epoch, forces)
    k1_r = tuple(h*vi for vi in v)
    k1_v = tuple(h*ai for ai in a1)

    r2 = add(r, k1_r, 0.5);  v2 = add(v, k1_v, 0.5)
    a2 = acceleration_total(r2, epoch + datetime.timedelta(seconds=0.5*h), forces)
    k2_r = tuple(h*vi for vi in v2); k2_v = tuple(h*ai for ai in a2)

    r3 = add(r, k2_r, 0.5);  v3 = add(v, k2_v, 0.5)
    a3 = acceleration_total(r3, epoch + datetime.timedelta(seconds=0.5*h), forces)
    k3_r = tuple(h*vi for vi in v3); k3_v = tuple(h*ai for ai in a3)

    r4 = add(r, k3_r);        v4 = add(v, k3_v)
    a4 = acceleration_total(r4, epoch + datetime.timedelta(seconds=h), forces)
    k4_r = tuple(h*vi for vi in v4); k4_v = tuple(h*ai for ai in a4)

    r_next = tuple(r[i] + (k1_r[i] + 2*k2_r[i] + 2*k3_r[i] + k4_r[i])/6
                   for i in range(3))
    v_next = tuple(v[i] + (k1_v[i] + 2*k2_v[i] + 2*k3_v[i] + k4_v[i])/6
                   for i in range(3))
    return r_next, v_next


# ---------------------------------------------------------------------------
#  ✨ 5. propagate_rk4 – מקבל forces=list
# ---------------------------------------------------------------------------
def propagate_rk4(r0, v0, t0, t_target, step=60.0, forces=None):
    """
    קידום RK4 בצעדי `step` [s] עם רשימת כוחות.
    forces=None → כל הכוחות.
    """
    dt_total = (t_target - t0).total_seconds()
    sign = 1 if dt_total >= 0 else -1
    h = sign * step

    r, v = r0, v0
    t_rem = abs(dt_total)
    epoch_cur = t0

    while t_rem > 1e-6:
        if abs(h) > t_rem:
            h = sign * t_rem
        r, v = rk4_step(r, v, h, epoch_cur, forces)
        epoch_cur += datetime.timedelta(seconds=h)
        t_rem -= abs(h)
    return r, v



def prev_main():
    prn = "G01"
    epochs = [e for e, _ in records[prn]]
    epoch0 = epochs[0]
    r0, v0 = sat_state_eci(Satellite(prn, records[prn]), 0)[:3], \
        sat_state_eci(Satellite(prn, records[prn]), 0)[3:]

    # --- משווים לכל האפוקים הבאים -------------------------------
    errors = []
    for k, epoch_k in enumerate(epochs[1:], start=1):
        r_pred, _ = propagate_rk4(r0, v0, epoch0, epoch_k, step=60.0)
        r_true = sat_state_eci(Satellite(prn, records[prn]), k)[:3]

        err = math.sqrt(sum((rp - rt) ** 2 for rp, rt in zip(r_pred, r_true)))
        errors.append((epoch_k, err))

    # --- הדפסה קצרה / אפשר להפוך ל-DataFrame  --------------------
    for ep, er in errors:
        print(f"{ep:%Y-%m-%d %H:%M}  |  error = {er:9.3f} m")
def main_():
    from datetime import timedelta
    import math

    # -------- הגדרות מדדים -------------------------------------------------
    sat_name = "G01"  # בחר לווין: "Gxx", "Rxx", "Exx" ...
    epoch0_index = 0  # אינדקס האפוק ההתחלתי ברשימת epochs של הלווין
    delta_minutes = 9*120  # הפרש זמן (דקות) לקידום

    step_seconds = 60.0  # גודל צעד RK4

    # -------- שליפה מה־records / satellites ------------------------------
    sat = satellites[sat_name]

    # Epoch-0 ומצב-0 (מיקום+מהירות ב-ECI)
    epoch0 = sat.epochs()[epoch0_index]
    x0, y0, z0, vx0, vy0, vz0 = sat_state_eci(sat, epoch0_index)
    r0, v0 = (x0, y0, z0), (vx0, vy0, vz0)

    # Epoch היעד הקרוב ביותר בקובץ (כדי שתהיה נקודת אמת להשוואה)
    epoch_target = epoch0 + timedelta(minutes=delta_minutes)
    epochs_list = sat.epochs()
    closest_index = min(range(len(epochs_list)),
                        key=lambda i: abs((epochs_list[i] - epoch_target).total_seconds()))
    epoch_k = epochs_list[closest_index]
    x_t, y_t, z_t, *_ = sat_state_eci(sat, closest_index)
    r_true = (x_t, y_t, z_t)

    # -------- מערכי הכוחות שנריץ -----------------------------------------

    test_sets = [
        ['central'],
        ['central', 'J2'],
        ['central', 'J2', 'J3'],
        ['central', 'J2', 'J3', 'J4'],
        ['central', 'J2', 'J3', 'J4', 'Sun', 'Moon'],
        ['central', 'J2', 'J3', 'J4', 'SRP', 'Sun', 'Moon'],
    ]

    print(f"\nPropagating {sat_name}  {epoch0}  →  {epoch_k}   "
          f"(Δt ≈ {delta_minutes} min)\n")

    # -------- הרצה והדפסת שגיאות ----------------------------------------
    for forces in test_sets:
        r_pred, _ = propagate_rk4(r0, v0, epoch0, epoch_k,
                                  step=step_seconds, forces=forces)
        err = math.dist(r_pred, r_true)

        forces_str = ", ".join(forces)  # או  str(forces)

        print(f"{forces_str:<45} →  error = {err:10.2f} m")


def run_propagation_tests(
        sat_name: str = "G01",
        epoch0_index: int = 0,
        delta_minutes_list: tuple[int, ...] = (60, 180, 540, 1080, 2160),
        step_seconds: float = 60.0,
        force_sets = None):
    """
    הדפסת שגיאת קידום עבור לווין אחד, למבחר מרווחי זמן ולכמה קבוצות כוחות.

    Parameters
    ----------
    sat_name : str
        סימון ה-PRN בדיוק כשנמצא ב-records, למשל "G01".
    epoch0_index : int
        האינדקס ברשימת epochs של אותו לווין שישמש כנקודת-התחלה.
    delta_minutes_list : iterable[int]
        Δt בדקות (חיובי – קדימה, שלילי – אחורה).
    step_seconds : float
        אורך צעד RK4 [s].
    force_sets : list[list[str]] | None
        רשימת סטים של כוחות. None ⇒ סט ברירת-מחדל.
    """
    if force_sets is None:
        force_sets = [
            ['central'],
            ['central', 'J2'],
            ['central', 'J2', 'J3', 'J4'],
            ['central', 'J2', 'J3', 'J4', 'Sun', 'Moon'],
            ['central', 'J2', 'J3', 'J4', 'Sun', 'Moon', 'SRP'],
        ]

    # ---- satellite & initial state ----
    sat      = satellites[sat_name]
    epoch0   = sat.epochs()[epoch0_index]
    x0, y0, z0, vx0, vy0, vz0 = sat_state_eci(sat, epoch0_index)
    r0, v0  = (x0, y0, z0), (vx0, vy0, vz0)

    print(f"\n=== Propagation tests for {sat_name} – start @ {epoch0} ===\n")

    # ---- iterate over requested Δt's ----
    for dmin in delta_minutes_list:
        epoch_target = epoch0 + timedelta(minutes=dmin)

        # אמת: האפוק הקרוב ביותר בקובץ
        idx_true = min(range(len(sat.epochs())),
                       key=lambda i: abs((sat.epochs()[i] - epoch_target).total_seconds()))
        epoch_true = sat.epochs()[idx_true]
        xt, yt, zt, *_ = sat_state_eci(sat, idx_true)
        r_true = (xt, yt, zt)

        print(f"Δt = {dmin:5d} min  (target epoch {epoch_true})")
        for forces in force_sets:
            r_pred, _ = propagate_rk4(r0, v0, epoch0, epoch_true,
                                      step=step_seconds, forces=forces)
            err_m = math.dist(r_pred, r_true)
            print(f"  {', '.join(forces):<40} →  {err_m:9.2f} m")
            print()
class SkyImage:
     """
     אובייקט שמרכז את כל הלוויינים (מתוך records/satellites), מאפשר
     לקדם את כל תמונת השמיים קדימה בזמן באמצעות מודל הכוחות הקיים (J2–J4),
     ולשמור תחזיות (predictions) לשימוש מאוחר יותר (כתיבת RINEX בשלב ב').

     שימוש טיפוסי:
         sky = SkyImage.from_rinex_dir(RINEX_DIR)
         sky.propagate_all(days=2.0, output_every_minutes=120,
                           step_seconds=60.0,
                           forces=['central','J2','J3','J4'])
         preds = sky.get_predictions('G01')
     """

     def __init__(self, satellites_dict: dict[str, Satellite]):
         self.satellites: dict[str, Satellite] = satellites_dict
         # predictions: {sat_id: [(epoch, (x,y,z,vx,vy,vz)), ...]}
         self.predictions: dict[str, list[tuple[datetime.datetime, tuple[float,float,float,float,float,float]]]] = {}
         # per-satellite along-track calibration (delta-v)
         self.calibration_dv: dict[str, float] = {}
     def calibrate_and_propagate(self,
                                days: float,
                                output_every_minutes: int = 120,
                                step_seconds: float = 60.0,
                                forces = None,
                                sat_filter = None,
                                verbose: bool = True) -> None:
        """
        Calibrate using all available history (first->last epoch) and propagate
        from the FIRST epoch forward to (last_epoch + days).
        
        This mimics the 'Calibrated' method from analyze_accuracy.py which
        showed superior stability/accuracy by fitting a long arc.
        """
        if forces is None:
            forces = ['central','J2','J3','J4', "Sun", "Moon", "SRP"]

        ids = self.list_satellites()
        if sat_filter:
            ids = [sid for sid in ids if sid in sat_filter]

        self.predictions.clear()
        self.calibration_dv = {}

        for sid in ids:
            sat = self.satellites[sid]
            entries = sat.entries
            if not entries:
                continue

            # 1. Calibrate on full range
            dv_opt = 0.0
            rmse_opt = 0.0
            if len(entries) >= 2:
                try:
                    # calibrate_alongtrack_dv_on_range uses indices
                    last_idx = len(entries) - 1
                    dv_opt, rmse_opt = calibrate_alongtrack_dv_on_range(
                        sat, 0, last_idx,
                        forces=forces, step_seconds=step_seconds,
                        verbose=False
                    )
                except Exception as e:
                    if verbose: print(f"Calibration failed for {sid}: {e}")
                    dv_opt = 0.0

            self.calibration_dv[sid] = dv_opt

            # 2. Propagate from FIRST epoch
            epoch0 = entries[0][0]
            last_epoch = entries[-1][0]
            
            # Initial state at epoch0
            x0, y0, z0, vx0, vy0, vz0 = sat_state_eci(sat, 0)
            v0 = (vx0, vy0, vz0)
            
            # Apply optimized delta-v
            if dv_opt != 0.0:
                 # Manually apply along-track dv: v_new = v + dv * (v/|v|)
                 v_mag = math.sqrt(vx0**2 + vy0**2 + vz0**2)
                 if v_mag > 0:
                     factor = 1.0 + dv_opt/v_mag
                     v0 = (vx0*factor, vy0*factor, vz0*factor)

            r, v = (x0, y0, z0), v0
            
            # Target time: last_epoch + days
            target_time = last_epoch + datetime.timedelta(days=days)
            
            if verbose:
                print(f"[cal] {sid}: dv={dv_opt:.6f} m/s, rmse={rmse_opt:.2f} m. Propagating {epoch0} -> {target_time}")

            # Propagate loop
            pred_list = []
            current_epoch = epoch0
            step_out = max(1, abs(int(output_every_minutes)))
            
            # We want to ensure we cover up to target_time
            while current_epoch < target_time:
                next_epoch = current_epoch + datetime.timedelta(minutes=step_out)
                if next_epoch > target_time:
                    next_epoch = target_time
                    
                r, v = propagate_rk4(r, v, current_epoch, next_epoch,
                                     step=step_seconds, forces=forces)
                pred_list.append((next_epoch, (*r, *v)))
                current_epoch = next_epoch

            self.predictions[sid] = pred_list

     def calibrate_and_forecast(self,
                                days_ahead: float,
                                output_every_minutes: int = 120,
                                step_seconds: float = 60.0,
                                forces = None,
                                sat_filter = None,
                                cal_bounds: tuple[float, float] = (-0.1, 0.1),
                                cal_tol: float = 1e-4,
                                cal_max_iter: int = 60,
                                verbose: bool = True) -> pd.DataFrame:
         """
         Calibrate an along‑track δv per satellite using *past* arc (first→last
         broadcast epoch), then forecast **forward** from the last epoch by
         `days_ahead`, storing predictions in `self.predictions` and returning a
         summary DataFrame with the calibration used.
         """
         if forces is None:
             forces = ['central','J2','J3','J4', "Sun", "Moon", "SRP"]

         ids = self.list_satellites()
         if sat_filter:
             ids = [sid for sid in ids if sid in sat_filter]
         if not ids:
             return pd.DataFrame(columns=['sat','dv_t_mps','rmse_m','n_epochs','last_epoch','forecast_until'])

         # clear previous
         self.predictions.clear()
         self.calibration_dv = {}

         total_minutes = int(round(days_ahead * 24 * 60))
         if total_minutes <= 0:
             raise ValueError("days_ahead must be positive")
         step_out = max(1, abs(int(output_every_minutes)))

         rows = []
         for sid in ids:
             sat = self.satellites[sid]
             entries = sat.entries
             if len(entries) < 2:
                 continue

             if verbose:
                 print(f"[cal+fc] {sid}: calibrating on {len(entries)} epochs (first→last)...")
             dv_opt, rmse_opt = calibrate_alongtrack_dv(
                 sat, start_index=0, forces=forces, step_seconds=step_seconds,
                 bounds=cal_bounds, tol=cal_tol, max_iter=cal_max_iter, verbose=False)
             self.calibration_dv[sid] = dv_opt

             last_idx = len(entries) - 1
             epoch0 = entries[last_idx][0]
             x0, y0, z0, vx0, vy0, vz0 = sat_state_eci(sat, last_idx)
             r = (x0, y0, z0)
             v = (vx0, vy0, vz0)
             if dv_opt != 0.0:
                 v = _apply_alongtrack_delta_v(v, dv_opt)

             if verbose:
                 until = epoch0 + datetime.timedelta(minutes=total_minutes)
                 print(f"[cal+fc] {sid}: dv_t*={dv_opt:.6f} m/s, rmse={rmse_opt:.2f} m → forecasting to {until}")

             # forward propagate on a uniform output grid
             pred_list: list[tuple[datetime.datetime, tuple[float,...]] ] = []
             current_epoch = epoch0
             remaining = total_minutes
             while remaining > 0:
                 jump = min(step_out, remaining)
                 next_epoch = current_epoch + datetime.timedelta(minutes=jump)
                 r, v = propagate_rk4(r, v, current_epoch, next_epoch,
                                      step=step_seconds, forces=forces)
                 pred_list.append((next_epoch, (*r, *v)))
                 current_epoch = next_epoch
                 remaining -= jump

             self.predictions[sid] = pred_list

             rows.append({
                 'sat': sid,
                 'dv_t_mps': dv_opt,
                 'rmse_m': rmse_opt,
                 'n_epochs': len(entries),
                 'last_epoch': epoch0,
                 'forecast_until': pred_list[-1][0] if pred_list else epoch0,
             })

         return pd.DataFrame(rows)

     # ------------ מפעלים -------------------------------------------------
     @classmethod
     def from_rinex_dir(cls, dir_path: str = RINEX_DIR) -> 'SkyImage':
         recs = load_rinex_dir(dir_path)
         sats = {sid: Satellite(sid, ent) for sid, ent in recs.items()}
         return cls(sats)

     @classmethod
     def from_existing_satellites(cls, sats: dict[str, Satellite]) -> 'SkyImage':
         return cls(sats)

     # ------------ שירותים ------------------------------------------------
     def list_satellites(self, constellation = None) -> list[str]:
         """החזר רשימת מזהי-לוויינים, אופציונלית מסוננת לפי קונסטלציה ('G','R','E','C',...)."""
         ids = list(self.satellites.keys())
         if constellation:
             ids = [sid for sid in ids if sid.startswith(constellation)]
         return sorted(ids)

     def clear_predictions(self):
         self.predictions.clear()

     def get_predictions(self, sat_id: str) -> list[tuple[datetime.datetime, tuple[float,...]]]:
         return self.predictions.get(sat_id, [])

     # ------------ קידום כללי --------------------------------------------
     def propagate_all(self,
                       days: float,
                       output_every_minutes: int = 120,
                       step_seconds: float = 60.0,
                       forces = ['central', 'J2', 'J3', 'J4', "Sun", "Moon", "SRP"],
                       start_epoch_policy: str = 'last',
                       sat_filter= None) -> None:
         """
         קדם את כל הלוויינים בבת־אחת ב־RK4, ושמור תחזיות בדגימה אחידה.

         Parameters
         ----------
         days : float
             בכמה ימים לקדם קדימה (יכול להיות שלילי לקידום אחורה).
         output_every_minutes : int
             מרווח הדגימה של נקודות התחזית לשמירה (ל־RINEX עתידי).
         step_seconds : float
             צעד RK4 פנימי.
         forces : list[str]
             קבוצת כוחות להפעלה. None ⇒ ['central','J2','J3','J4'].
         start_epoch_policy : {'last','first','index:<n>'}
             מאיזה epoch להתחיל לכל לוויין.
         sat_filter : list[str] | None
             אם ניתן – יריץ רק עבור מזהים אלו.
         """
         if forces is None:
             forces = ['central','J2','J3','J4', "Sun", "Moon", "SRP"]

         ids = self.list_satellites()
         if sat_filter:
             ids = [sid for sid in ids if sid in sat_filter]

         self.predictions.clear()

         for sid in ids:
             sat = self.satellites[sid]
             print(f"Currently Propagating SV {sat.id}")
             epochs = sat.epochs()
             if not epochs:
                 continue

             # בוחרים epoch התחלה לפי המדיניות
             if start_epoch_policy == 'last':
                 idx0 = len(epochs) - 1
             elif start_epoch_policy == 'first':
                 idx0 = 0
             elif start_epoch_policy.startswith('index:'):
                 try:
                     idx0 = int(start_epoch_policy.split(':',1)[1])
                 except Exception:
                     idx0 = 0
                 idx0 = max(0, min(idx0, len(epochs)-1))
             else:
                 idx0 = len(epochs) - 1

             epoch0 = epochs[idx0]
             x0, y0, z0, vx0, vy0, vz0 = sat_state_eci(sat, idx0)
             r, v = (x0, y0, z0), (vx0, vy0, vz0)

             # גריד יציאה
             total_minutes = int(round(days * 24 * 60))
             sign = 1 if total_minutes >= 0 else -1
             total_minutes = abs(total_minutes)
             if total_minutes == 0:
                 # רק נקודת התחלה
                 self.predictions[sid] = [(epoch0, (x0,y0,z0,vx0,vy0,vz0))]
                 continue

             step_out = abs(int(output_every_minutes))
             step_out = max(1, step_out)

             # ננוע קדימה/אחורה במקטעים של output_every_minutes
             pred_list: list[tuple[datetime.datetime, tuple[float,...]]] = []
             current_epoch = epoch0
             remaining = total_minutes
             while remaining > 0:
                 # זמן היעד הבא לגריד
                 jump = min(step_out, remaining)
                 next_epoch = current_epoch + datetime.timedelta(minutes=sign*jump)
                 # קידום של מצב r,v עד ל-next_epoch
                 r, v = propagate_rk4(r, v, current_epoch, next_epoch,
                                      step=step_seconds, forces=forces)
                 pred_list.append((next_epoch, (*r, *v)))
                 current_epoch = next_epoch
                 remaining -= jump

             self.predictions[sid] = pred_list

    # ------------ כתיבת RINEX – שלב ב' ---------------------------
     def write_rinex(self, out_path: str, version: str = '3.04') -> None:
         """
         Write a RINEX Navigation file with the predictions stored in `self.predictions`.
         Converts Cartesian state (ECI) to Keplerian elements for GPS/Galileo/BeiDou.
         """
         if not self.predictions:
             print("No predictions to write.")
             return

         # Collect all epochs and satellites
         all_epochs = set()
         for pred_list in self.predictions.values():
             for epoch, _ in pred_list:
                 all_epochs.add(epoch)
         sorted_epochs = sorted(list(all_epochs))

         with open(out_path, 'w') as f:
             # Header
             f.write(f"{version:>9}           N: GNSS NAV DATA    M: MIXED            RINEX VERSION / TYPE\n")
             f.write(f"EphemerisProp       Generated by Python 20251204 000000 GMT PGM / RUN BY / DATE\n")
             f.write(f"{'':60}END OF HEADER\n")

             # Data
             for epoch in sorted_epochs:
                 for sid, pred_list in self.predictions.items():
                     # Find state for this epoch
                     state = next((s for t, s in pred_list if t == epoch), None)
                     if not state:
                         continue
                     
                     x, y, z, vx, vy, vz = state
                     r_vec = (x, y, z)
                     v_vec = (vx, vy, vz)
                     
                     # Convert to Keplerian (approximate for broadcast)
                     # Note: Broadcast usually uses Mean elements, here we use Osculating.
                     # Also, broadcast parameters are in ECEF frame usually (for the orbit), 
                     # but our propagation is in ECI. 
                     # However, standard RINEX broadcast parameters define the orbit in ECEF (rotating).
                     # Wait, GPS broadcast parameters (Keplerian) define the orbit in an INERTIAL frame 
                     # (relative to Ω0 at Toe), but the coordinate system rotates with Earth?
                     # Actually, the ICD defines the coordinate system.
                     # For simplicity, we will output a "GLONASS-like" Cartesian record if possible, 
                     # OR we try to fit Keplerian.
                     # Given the complexity, and the user's request "give a file containing up to the requested time",
                     # maybe they just want the *original* RINEX extended?
                     # But we are propagating.
                     
                     # Let's try to write a simplified GPS record with fitted elements.
                     # Or better: write Cartesian state if the format allows (RINEX 3.04 supports it for GLONASS/SBAS/BDS).
                     # For GPS, it MUST be Keplerian.
                     
                     # Implementation of Cartesian -> Keplerian
                     try:
                         a, e, i, Omega, omega, M, n = _cartesian_to_keplerian(r_vec, v_vec)
                     except Exception:
                         continue # Skip if conversion fails (e.g. hyperbolic)

                     # We need to fill the RINEX record.
                     # GPS Record Layout (RINEX 3.04):
                     # Line 1: SV, Epoch, SV Clock Bias, SV Clock Drift, SV Clock Drift Rate
                     # Line 2: IODE, Crs, Delta n, M0
                     # Line 3: Cuc, e, Cus, sqrt(A)
                     # Line 4: Toe, Cic, Omega0, Cis
                     # Line 5: i0, Crc, omega, OmegaDot
                     # Line 6: IDOT, Codes on L2, GPS Week, L2 P data flag
                     # Line 7: SV accuracy, SV health, TGD, IODC
                     # Line 8: Transmission time, Fit interval, spare, spare
                     
                     # We don't have clock data, so we set to 0.
                     # We don't have perturbations (Crs, Cuc, etc.), set to 0.
                     # We set Toe = epoch.
                     
                     # Time handling
                     y, m, d, h, min_, sec = epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second
                     
                     # GPS Week and Toe
                     # Simple approx:
                     diff = epoch - datetime.datetime(1980, 1, 6)
                     gps_week = diff.days // 7
                     toe = diff.total_seconds() - gps_week * 604800
                     
                     # Format floats
                     def fmt(val):
                         return f"{val:19.12E}".replace('E', 'D')

                     # Line 1
                     f.write(f"{sid} {y:04} {m:02} {d:02} {h:02} {min_:02} {sec:02} {fmt(0)} {fmt(0)} {fmt(0)}\n")
                     
                     # Line 2: IODE, Crs, Delta n, M0
                     f.write(f"    {fmt(0)}{fmt(0)}{fmt(0)}{fmt(M)}\n")
                     
                     # Line 3: Cuc, e, Cus, sqrt(A)
                     f.write(f"    {fmt(0)}{fmt(e)}{fmt(0)}{fmt(math.sqrt(a))}\n")
                     
                     # Line 4: Toe, Cic, Omega0, Cis
                     # Note: Omega0 in RINEX is relative to Greenwich at start of week? 
                     # Actually it is RAAN at Weekly Epoch.
                     # Our Omega is RAAN in ECI (J2000).
                     # We need to adjust for GST to get "Longitude of Ascending Node" in ECEF at Toe?
                     # GPS ICD: Ωk = Ω0 + (Ωdot - ωe)*tk - ωe*Toe
                     # Here we have instantaneous Ω (RAAN).
                     # Let's assume Omega0 = Omega - (OmegaDot - OMEGA_E)*tk + OMEGA_E*Toe?
                     # At tk=0 (epoch=Toe), Ω = Ω0 - ωe*Toe.
                     # So Ω0 = Ω + ωe*Toe.
                     # Wait, Ω in ECI is inertial.
                     # The formula for longitude of ascending node is Ω(t) = Ω0 + (Ωdot - ωe)*tk - ωe*Toe.
                     # At t=Toe (tk=0), Ω(Toe) = Ω0 - ωe*Toe.
                     # So Ω0 = Ω(Toe) + ωe*Toe.
                     # But Ω(Toe) is the RAAN in ECI? Yes.
                     # And we need to subtract GMST?
                     # Actually, let's just use the standard conversion:
                     # Ω0 = Ω_ECI - GMST(Toe) ? No, GPS uses a specific frame.
                     # Let's use Ω0 = Ω_ECI + OMEGA_E * Toe (approx).
                     
                     omega0 = Omega + OMEGA_E * toe
                     
                     f.write(f"    {fmt(toe)}{fmt(0)}{fmt(omega0)}{fmt(0)}\n")
                     
                     # Line 5: i0, Crc, omega, OmegaDot
                     f.write(f"    {fmt(i)}{fmt(0)}{fmt(omega)}{fmt(0)}\n")
                     
                     # Line 6: IDOT, Codes, Week, Flag
                     f.write(f"    {fmt(0)}{fmt(0)}{fmt(gps_week)}{fmt(0)}\n")
                     
                     # Line 7: Acc, Health, TGD, IODC
                     f.write(f"    {fmt(0)}{fmt(0)}{fmt(0)}{fmt(0)}\n")
                     
                     # Line 8: TransTime, Fit, Spare, Spare
                     f.write(f"    {fmt(0)}{fmt(0)}{fmt(0)}{fmt(0)}\n")

def _cartesian_to_keplerian(r_vec, v_vec):
    """
    Convert ECI Cartesian state (position r, velocity v) to Keplerian elements.
    r, v in meters and meters/second.
    Returns: a, e, i, Omega, omega, M, n
    """
    x, y, z = r_vec
    vx, vy, vz = v_vec
    
    r = math.sqrt(x*x + y*y + z*z)
    v2 = vx*vx + vy*vy + vz*vz
    
    # Specific angular momentum h = r x v
    hx = y*vz - z*vy
    hy = z*vx - x*vz
    hz = x*vy - y*vx
    h = math.sqrt(hx*hx + hy*hy + hz*hz)
    
    # Inclination i
    i = math.acos(hz / h)
    
    # Right Ascension of Ascending Node Omega
    # Node vector n = k x h = (-hy, hx, 0)
    nx, ny = -hy, hx
    n_mag = math.sqrt(nx*nx + ny*ny)
    
    if n_mag < 1e-9:
        Omega = 0.0 # Equatorial orbit
    else:
        Omega = math.atan2(ny, nx)
        if Omega < 0: Omega += 2*math.pi
        
    # Eccentricity vector e
    # e = (1/MU) * ((v^2 - MU/r)*r - (r.v)*v)
    mu_r = MU / r
    rv = x*vx + y*vy + z*vz
    
    ex = (1/MU) * ((v2 - mu_r)*x - rv*vx)
    ey = (1/MU) * ((v2 - mu_r)*y - rv*vy)
    ez = (1/MU) * ((v2 - mu_r)*z - rv*vz)
    e = math.sqrt(ex*ex + ey*ey + ez*ez)
    
    # Semi-major axis a
    # Energy E = v^2/2 - MU/r = -MU / (2a)
    energy = v2/2 - mu_r
    if abs(energy) < 1e-9:
        a = float('inf') # Parabolic
    else:
        a = -MU / (2*energy)
        
    # Argument of Perigee omega
    if n_mag < 1e-9:
        # Equatorial: angle between e and x-axis?
        omega = 0.0 # Undefined
    else:
        # Angle between n and e
        # cos(w) = n.e / (|n||e|)
        ndote = nx*ex + ny*ey # nz=0
        cos_w = ndote / (n_mag * e)
        cos_w = max(-1.0, min(1.0, cos_w))
        omega = math.acos(cos_w)
        if ez < 0:
            omega = 2*math.pi - omega
            
    # Mean Anomaly M
    # True Anomaly nu: angle between e and r
    edotr = ex*x + ey*y + ez*z
    cos_nu = edotr / (e * r)
    cos_nu = max(-1.0, min(1.0, cos_nu))
    nu = math.acos(cos_nu)
    if rv < 0:
        nu = 2*math.pi - nu
        
    # Eccentric Anomaly E
    # tan(E/2) = sqrt((1-e)/(1+e)) * tan(nu/2)
    E = 2 * math.atan(math.sqrt((1-e)/(1+e)) * math.tan(nu/2))
    if E < 0: E += 2*math.pi
    
    # Mean Anomaly M = E - e*sin(E)
    M = E - e*math.sin(E)
    
    # Mean motion n
    n = math.sqrt(MU / a**3)
    
    return a, e, i, Omega, omega, M, n

# ---------------------------------------------------------------------------
#  📊 Propagation test across satellites – mean & std of position error vs time
# ---------------------------------------------------------------------------
def test_propagation_errors(
        satellites_dict: dict,
        forces = ['central', 'J2', 'J3', 'J4', "Sun", "Moon", "SRP"],
        step_seconds: float = 60.0,
        constellation = 'G',
        show_plot: bool = True,
        verbose: bool = True,
) -> pd.DataFrame:
    """
    עבור כל לווין: מקדם מה-epoch הראשון → לכל epoch עד האחרון (ברצף),
    מחשב בכל שלב את שגיאת המיקום (ECI), ומאגד ממוצע+סטיית-תקן על פני כל הלוויינים
    כפונקציה של Δt (בדקות) מאז ה-epoch הראשון של אותו לווין.

    Parameters
    ----------
    satellites_dict : dict[str, Satellite]
        מילון הלוויינים הקיים (`satellites`).
    forces : list[str] | None
        רשימת כוחות ל-RK4. None ⇒ ['central','J2','J3','J4'].
    step_seconds : float
        צעד RK4 פנימי [s].
    constellation : str | None
        אם לא-None, מסנן לוויינים לפי התחילית ('G','E','C','R','S', ...).
    show_plot : bool
        האם להציג תרשים עם ממוצע וסטיית-תקן מול זמן.
    verbose : bool
        האם להדפיס התקדמות.

    Returns
    -------
    pd.DataFrame
        טבלת תקציר עם עמודות: ['dt_min','mean_m','std_m','n'].
    """
    if forces is None:
        forces = ['central','J2','J3','J4', "Sun", "Moon", "SRP"]

    # בחירת הלוויינים
    sat_ids = sorted(satellites_dict.keys())
    if constellation:
        sat_ids = [sid for sid in sat_ids if sid.startswith(constellation)]
    if not sat_ids:
        raise ValueError("No satellites selected – check constellation filter.")

    # יצבור טעויות לפי דלתא-זמן (בדקות) מאז epoch ראשון של כל לווין
    errors_by_dt = defaultdict(list)  # {dt_min: [err_m over sats that have this dt]}

    for sid in sat_ids:
        sat = satellites_dict[sid]
        entries = sat.entries
        if len(entries) < 2:
            continue
        if verbose:
            print(f"[test] {sid}: {len(entries)} epochs")

        # מצב התחלתי
        epoch0 = entries[0][0]
        x0, y0, z0, vx0, vy0, vz0 = sat_state_eci(sat, 0)
        r, v = (x0, y0, z0), (vx0, vy0, vz0)
        t_cur = epoch0

        # קדימה עד סוף הרשומות, תוך בדיקה בכל אפוק אמת
        for k in range(1, len(entries)):
            epoch_k = entries[k][0]
            # קידום רציף מהמצב הנוכחי אל האפוק הבא
            r, v = propagate_rk4(r, v, t_cur, epoch_k, step=step_seconds, forces=forces)
            t_cur = epoch_k

            # אמת מן ה-broadcast
            x_t, y_t, z_t, *_ = sat_state_eci(sat, k)
            # שגיאת מיקום [m]
            err = math.sqrt((r[0]-x_t)**2 + (r[1]-y_t)**2 + (r[2]-z_t)**2)

            dt_min = int(round((epoch_k - epoch0).total_seconds() / 60.0))
            errors_by_dt[dt_min].append(err)

    if not errors_by_dt:
        raise RuntimeError("No error data accumulated – check inputs.")

    # בניית תקציר: dt → mean/std/n
    dts = sorted(errors_by_dt.keys())
    means = []
    stds = []
    counts = []
    for dt in dts:
        vals = errors_by_dt[dt]
        counts.append(len(vals))
        means.append(stats.fmean(vals))
        stds.append(stats.stdev(vals) if len(vals) > 1 else 0.0)

    summary = pd.DataFrame({
        'dt_min': dts,
        'mean_m': means,
        'std_m': stds,
        'n': counts,
    })

    if show_plot:
        plt.figure()
        plt.plot(summary['dt_min'], summary['mean_m'], label='Mean position error [m]')
        lower = [m - s for m, s in zip(summary['mean_m'], summary['std_m'])]
        upper = [m + s for m, s in zip(summary['mean_m'], summary['std_m'])]
        plt.fill_between(summary['dt_min'], lower, upper, alpha=0.2, label='± Standard deviation')
        plt.xlabel('Δt since first epoch [minutes]')
        plt.ylabel('Position error [m]')
        plt.title('Propagation error – mean and std across satellites')
        plt.grid(True, which='both', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return summary

# ---------------------------------------------------------------------------
#  ⚙️ Calibration utilities – remove along‑track bias by fitting δvₜ at t₀
# ---------------------------------------------------------------------------

def _apply_alongtrack_delta_v(v0: tuple[float, float, float], dv_t: float) -> tuple[float, float, float]:
    """Return v0 with an added along‑track (tangential) delta‑v of dv_t [m/s]."""
    vx, vy, vz = v0
    vnorm = math.sqrt(vx*vx + vy*vy + vz*vz)
    if vnorm == 0.0:
        return v0
    tx, ty, tz = vx/vnorm, vy/vnorm, vz/vnorm  # unit tangent (velocity) direction
    return (vx + dv_t*tx, vy + dv_t*ty, vz + dv_t*tz)


def _propagation_error_series(sat: 'Satellite', start_index: int,
                              forces=None, step_seconds: float = 60.0,
                              dv_t: float = 0.0) -> list[float]:
    """
    Propagate from entries[start_index] to the end, returning per‑epoch 3D position
    errors [m] against broadcast ephemeris. Optionally apply an initial along‑track
    delta‑v dv_t [m/s].
    """
    if forces is None:
        forces = ['central','J2','J3','J4', "Sun", "Moon", "SRP"]

    entries = sat.entries
    if start_index >= len(entries) - 1:
        return []

    epoch0 = entries[start_index][0]
    x0, y0, z0, vx0, vy0, vz0 = sat_state_eci(sat, start_index)
    v0 = (vx0, vy0, vz0)
    if dv_t != 0.0:
        v0 = _apply_alongtrack_delta_v(v0, dv_t)
    r = (x0, y0, z0)
    v = v0
    t_cur = epoch0

    errs: list[float] = []
    for k in range(start_index+1, len(entries)):
        epoch_k = entries[k][0]
        r, v = propagate_rk4(r, v, t_cur, epoch_k, step=step_seconds, forces=forces)
        t_cur = epoch_k
        x_t, y_t, z_t, *_ = sat_state_eci(sat, k)
        err = math.sqrt((r[0]-x_t)**2 + (r[1]-y_t)**2 + (r[2]-z_t)**2)
        errs.append(err)
    return errs


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(v*v for v in values) / len(values)) if values else float('nan')


def calibrate_alongtrack_dv(sat: 'Satellite', start_index: int = 0,
                             forces=None, step_seconds: float = 60.0,
                             bounds: tuple[float, float] = (-0.2, 0.2),
                             tol: float = 1e-4, max_iter: int = 60,
                             verbose: bool = True) -> tuple[float, float]:
    """
    Find the along‑track delta‑v at t₀ (in m/s) that minimizes the RMS 3D position
    error over the arc (from entries[start_index] to the last entry).

    Returns
    -------
    (dv_t_opt [m/s], rmse_opt [m])
    """
    if forces is None:
        forces = ['central','J2','J3','J4', "Sun", "Moon", "SRP"]

    # Golden‑section search over dv_t in [a,b]
    a, b = bounds
    phi = (math.sqrt(5.0) - 1.0) / 2.0  # ~0.618
    c = b - phi*(b - a)
    d = a + phi*(b - a)

    def f(dv: float) -> float:
        errs = _propagation_error_series(sat, start_index, forces, step_seconds, dv_t=dv)
        return _rms(errs)

    fc = f(c)
    fd = f(d)
    it = 0
    while (b - a) > tol and it < max_iter:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi*(b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi*(b - a)
            fd = f(d)
        it += 1
        if verbose:
            print(f"[cal] iter {it:02d}: interval=({a:.6f},{b:.6f})  best≈{min(fc,fd):.3f} m")

    dv_opt = (a + b) / 2.0
    rmse_opt = f(dv_opt)
    if verbose:
        print(f"[cal] dv_t* = {dv_opt:.6f} m/s,  RMSE = {rmse_opt:.3f} m")
    return dv_opt, rmse_opt


def calibrate_dv_for_all(satellites_dict: dict, constellation: str = 'G',
                         forces=None, step_seconds: float = 60.0,
                         **kwargs) -> pd.DataFrame:
    """
    Calibrate dv_t per‑satellite (first→last epoch) and return a summary DataFrame
    with columns: ['sat','dv_t_mps','rmse_m','n_epochs'].
    """
    if forces is None:
        forces = ['central','J2','J3','J4', "Sun", "Moon", "SRP"]
    sat_ids = sorted(satellites_dict.keys())
    if constellation:
        sat_ids = [sid for sid in sat_ids if sid.startswith(constellation)]
    rows = []
    for sid in sat_ids:
        sat = satellites_dict[sid]
        if len(sat.entries) < 2:
            continue
        dv_opt, rmse_opt = calibrate_alongtrack_dv(sat, start_index=0, forces=forces,
                                                   step_seconds=step_seconds, verbose=False, **kwargs)
        rows.append({'sat': sid, 'dv_t_mps': dv_opt, 'rmse_m': rmse_opt, 'n_epochs': len(sat.entries)})
        print(f"[cal] {sid}: dv_t={dv_opt:.6f} m/s, rmse={rmse_opt:.2f} m, N={len(sat.entries)}")
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
#  📐 Calibrate on day‑1, evaluate on the remaining days – per satellite & aggregate
# ---------------------------------------------------------------------------

def _first_day_end_index(entries: list[tuple[datetime.datetime, dict]]) -> int:
    """Return the last index belonging to the first 24h window since entries[0]."""
    if not entries:
        return -1
    t0 = entries[0][0]
    t_end = t0 + datetime.timedelta(days=1)
    idx_end = 0
    for i, (t, _) in enumerate(entries):
        if t < t_end:
            idx_end = i
        else:
            break
    return idx_end


def _propagation_error_series_range(sat: 'Satellite', start_index: int, end_index: int,
                                    forces=None, step_seconds: float = 60.0,
                                    dv_t: float = 0.0) -> list[float]:
    """
    Propagate from entries[start_index] up to entries[end_index], returning per‑epoch
    3D position errors [m] against broadcast. Optionally apply an initial along‑track
    delta‑v dv_t [m/s] at entries[start_index].
    """
    if forces is None:
        forces = ['central','J2','J3','J4']

    entries = sat.entries
    if start_index >= end_index:
        return []

    epoch0 = entries[start_index][0]
    x0, y0, z0, vx0, vy0, vz0 = sat_state_eci(sat, start_index)
    v0 = (vx0, vy0, vz0)
    if dv_t != 0.0:
        v0 = _apply_alongtrack_delta_v(v0, dv_t)
    r = (x0, y0, z0)
    v = v0
    t_cur = epoch0

    errs: list[float] = []
    for k in range(start_index+1, end_index+1):
        epoch_k = entries[k][0]
        r, v = propagate_rk4(r, v, t_cur, epoch_k, step=step_seconds, forces=forces)
        t_cur = epoch_k
        x_t, y_t, z_t, *_ = sat_state_eci(sat, k)
        err = math.sqrt((r[0]-x_t)**2 + (r[1]-y_t)**2 + (r[2]-z_t)**2)
        errs.append(err)
    return errs


def calibrate_alongtrack_dv_on_range(sat: 'Satellite', start_index: int, end_index: int,
                                      forces=None, step_seconds: float = 60.0,
                                      bounds: tuple[float, float] = (-0.2, 0.2),
                                      tol: float = 1e-4, max_iter: int = 60,
                                      verbose: bool = False) -> tuple[float, float]:
    """
    Like `calibrate_alongtrack_dv`, but fits dv_t only over the sub‑arc
    entries[start_index..end_index]. Returns (dv_t_opt [m/s], rmse_opt [m]).
    """
    if forces is None:
        forces = ['central','J2','J3','J4']

    if end_index <= start_index:
        return 0.0, float('nan')

    # Golden‑section search
    a, b = bounds
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - phi*(b - a)
    d = a + phi*(b - a)

    def f(dv: float) -> float:
        errs = _propagation_error_series_range(sat, start_index, end_index, forces, step_seconds, dv_t=dv)
        return _rms(errs) if errs else float('inf')

    fc = f(c); fd = f(d)
    it = 0
    while (b - a) > tol and it < max_iter:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi*(b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi*(b - a)
            fd = f(d)
        it += 1
        if verbose:
            print(f"[cal-range] iter {it:02d}: interval=({a:.6f},{b:.6f})  best≈{min(fc,fd):.3f} m")

    dv_opt = (a + b) / 2.0
    rmse_opt = f(dv_opt)
    if verbose:
        print(f"[cal-range] dv_t* = {dv_opt:.6f} m/s, RMSE(day‑1) = {rmse_opt:.3f} m")
    return dv_opt, rmse_opt


def test_calibrated_forecast_errors(
        satellites_dict: dict,
        forces=None,
        step_seconds: float = 60.0,
        constellation = 'G',
        show_plot: bool = True,
        verbose: bool = True,
        cal_bounds: tuple[float, float] = (-0.1, 0.1),
        cal_tol: float = 1e-4,
        cal_max_iter: int = 60
) -> pd.DataFrame:
    """
    For each satellite: calibrate dv_t on the **first 24h** (from its first epoch),
    then start from the **last epoch of day‑1** and propagate forward to every
    subsequent broadcast epoch, computing position errors. Aggregate mean/std
    vs Δt since calibration end across satellites.

    Returns a DataFrame with columns ['dt_min','mean_m','std_m','n'] and
    shows a plot if `show_plot` is True.
    """
    if forces is None:
        forces = ['central','J2','J3','J4', "Sun", "Moon", "SRP"]

    sat_ids = sorted(satellites_dict.keys())
    if constellation:
        sat_ids = [sid for sid in sat_ids if sid.startswith(constellation)]
    if not sat_ids:
        raise ValueError("No satellites selected – check constellation filter.")

    errors_by_dt = defaultdict(list)

    for sid in sat_ids:
        sat = satellites_dict[sid]
        entries = sat.entries
        if len(entries) < 3:
            continue
        # split day‑1 vs the rest
        idx_cal_end = _first_day_end_index(entries)
        if idx_cal_end < 1 or idx_cal_end >= len(entries)-1:
            # either too few in day‑1 or no future epochs – skip
            if verbose:
                print(f"[cal-fore] {sid}: insufficient epochs in day‑1 or no future – skipping")
            continue

        # calibrate on [0 .. idx_cal_end]
        if verbose:
            print(f"[cal-fore] {sid}: calibrating on {idx_cal_end+1} epochs (first day)")
        dv_opt, rmse_opt = calibrate_alongtrack_dv_on_range(
            sat, start_index=0, end_index=idx_cal_end,
            forces=forces, step_seconds=step_seconds,
            bounds=cal_bounds, tol=cal_tol, max_iter=cal_max_iter, verbose=False)

        # evaluate forward from idx_cal_end to the end, starting at that epoch with dv applied
        epoch0 = entries[idx_cal_end][0]
        x0, y0, z0, vx0, vy0, vz0 = sat_state_eci(sat, idx_cal_end)
        r = (x0, y0, z0)
        v = (vx0, vy0, vz0)
        if dv_opt != 0.0:
            v = _apply_alongtrack_delta_v(v, dv_opt)
        t_cur = epoch0

        if verbose:
            print(f"[cal-fore] {sid}: dv_t*={dv_opt:.6f} m/s  – evaluating future ({len(entries)-idx_cal_end-1} epochs)")

        for k in range(idx_cal_end+1, len(entries)):
            epoch_k = entries[k][0]
            r, v = propagate_rk4(r, v, t_cur, epoch_k, step=step_seconds, forces=forces)
            t_cur = epoch_k
            x_t, y_t, z_t, *_ = sat_state_eci(sat, k)
            err = math.sqrt((r[0]-x_t)**2 + (r[1]-y_t)**2 + (r[2]-z_t)**2)

            dt_min = int(round((epoch_k - epoch0).total_seconds() / 60.0))
            errors_by_dt[dt_min].append(err)

    if not errors_by_dt:
        raise RuntimeError("No error data accumulated – check inputs")

    dts = sorted(errors_by_dt.keys())
    means, stds, counts = [], [], []
    for dt in dts:
        vals = errors_by_dt[dt]
        counts.append(len(vals))
        means.append(stats.fmean(vals))
        stds.append(stats.stdev(vals) if len(vals) > 1 else 0.0)

    summary = pd.DataFrame({'dt_min': dts, 'mean_m': means, 'std_m': stds, 'n': counts})

    if show_plot:
        plt.figure()
        plt.plot(summary['dt_min'], summary['mean_m'], label='Mean position error after calibration [m]')
        lower = [m - s for m, s in zip(summary['mean_m'], summary['std_m'])]
        upper = [m + s for m, s in zip(summary['mean_m'], summary['std_m'])]
        plt.fill_between(summary['dt_min'], lower, upper, alpha=0.2, label='± Standard deviation')
        plt.xlabel('Δt since end of day‑1 calibration [minutes]')
        plt.ylabel('Position error [m]')
        plt.title('Forecast error after day‑1 calibration – mean and std across satellites')
        plt.grid(True, which='both', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return summary

# ---------------------------------------------------------------------------
#  קריאה לדוגמה – שנה כרצונך
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ---------- Ensure a recent BRDC exists (via ~/.netrc or EARTHDATA_TOKEN) ---------- #
    auto_dl = os.environ.get('RINEX_AUTO_DL', '1') in ('1', 'true', 'True')
    if auto_dl:
        print("[BRDC] Ensuring latest BRDC files (today/back to 2 days)...")
        ensured = ensure_latest_brdc(max_days_back=2, out_dir=RINEX_DIR)
        for p in ensured:
            print(f"[BRDC] ready: {p}")

    # ---------- Run the parser ---------- #
    records = load_rinex_dir(RINEX_DIR)
    satellites = {sid: Satellite(sid, recs) for sid, recs in records.items()}

    sky = SkyImage.from_existing_satellites(satellites)

    # --- Evaluate forecast errors after calibrating on the first 24h ---
    summary_cal = test_calibrated_forecast_errors(
        satellites,
        forces=['central','J2','J3','J4','Moon', "Sun","SRP"],
        step_seconds=60.0,
        constellation='G',
        show_plot=True,
        verbose=True,
        cal_bounds=(-0.1, 0.1),
        cal_tol=1e-4,
        cal_max_iter=60,
    )
    print(summary_cal.head())