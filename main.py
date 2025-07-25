import datetime, math, pandas as pd
import os
import gzip
import glob
from math import sin, cos, sqrt, atan2, radians

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
    'I': [  # IRNSS – treat like GPS (7 data lines)
        'SV_clock_bias', 'SV_clock_drift', 'SV_clock_drift_rate'
    ] + [f'param_{k}' for k in range(29)],
    'J': [  # QZSS – same layout as GPS
        'SV_clock_bias', 'SV_clock_drift', 'SV_clock_drift_rate'
    ] + [f'param_{k}' for k in range(29)],
}

def _to_float(s: str) -> float:
    """Convert FORTRAN‐style float with D exponent to Python float."""
    return float(s.replace('D', 'E').strip()) if s.strip() else math.nan

def parse_rinex_nav(path: str):
    """Return {satellite_id: [(epoch_datetime, {param_name: value})]} dict."""
    data = {}
    with open(path, "r") as f:
        # -------- skip header -------- #
        for ln in f:
            if "END OF HEADER" in ln:
                break

        while True:
            hdr = f.readline()
            if not hdr:
                break
            if not hdr.strip():
                continue  # ignore blank lines

            # First 22 columns contain PRN & epoch fields
            parts = hdr[:22].split()
            sat = parts[0]          # e.g. G01, R06, E12 …
            yy, mm, dd, hh, mi, ss  = map(int, parts[1:7])
            epoch = datetime.datetime(yy, mm, dd, hh, mi, ss)

            # Three SV‑clock parameters sit at fixed cols 23‑41, 42‑60, 61‑79
            clk = [_to_float(hdr[c:c+19]) for c in (23, 42, 61)]

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

    if vx_ecef == vy_ecef == vz_ecef == 0.0:
        return (x_eci, y_eci, z_eci)           # רק מיקום

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

def sun_position_eci(epoch: datetime.datetime) -> tuple[float, float, float]:
    """ECI Sun vector *Earth-centred* (Skyfield • מדויק ~1 m)."""
    t = _ts.utc(epoch.year, epoch.month, epoch.day,
                epoch.hour, epoch.minute, epoch.second + epoch.microsecond/1e6)
    # Sun w.r.t. Earth
    pos = ( _eph['sun'].at(t) - earth.at(t) ).position.m  # ← יחידות [m]
    return tuple(pos)

def moon_position_eci(epoch: datetime.datetime) -> tuple[float, float, float]:
    """ECI Moon vector *Earth-centred* (Skyfield)."""
    t = _ts.utc(epoch.year, epoch.month, epoch.day,
                epoch.hour, epoch.minute, epoch.second + epoch.microsecond/1e6)
    pos = ( _eph['moon'].at(t) - earth.at(t) ).position.m
    return tuple(pos)


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
    תאוצת Solar Radiation Pressure (שטח/מסה יחסי + Cr).
    לחץ קרינתי: 4.56e-6 N/m² ב-1 AU.
    """
    dx = r_sat[0] - r_sun[0]
    dy = r_sat[1] - r_sun[1]
    dz = r_sat[2] - r_sun[2]
    d2 = dx*dx + dy*dy + dz*dz
    d  = math.sqrt(d2)

    P0 = 4.56e-6           # N/m²
    accel = P0 * Cr * A_mass * (AU / d)**2   # [m/s²]

    ax = -accel * dx / d
    ay = -accel * dy / d
    az = -accel * dz / d
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
    active=None  →  כל הכוחות ברשימה.
    """
    if active is None:
        active = FORCES.keys()

    ax = ay = az = 0.0
    for key in active:
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


# ---------- Run the parser ---------- #
records = load_rinex_dir(RINEX_DIR)
satellites = {sid: Satellite(sid, recs) for sid, recs in records.items()}


def run_propagation_tests(
        sat_name: str = "G01",
        epoch0_index: int = 0,
        delta_minutes_list: tuple[int, ...] = (60, 180, 540, 1080, 2160),
        step_seconds: float = 60.0,
        force_sets = None,
):
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
    from datetime import timedelta
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

# ---------------------------------------------------------------------------
#  קריאה לדוגמה – שנה כרצונך
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_propagation_tests(
        sat_name="G01",         # לווין GPS לדוגמה
        epoch0_index=0,         # אפוק התחלתי
        delta_minutes_list=(24*60, 48*60, 72*60),
        step_seconds=60.0       # צעד RK
    )