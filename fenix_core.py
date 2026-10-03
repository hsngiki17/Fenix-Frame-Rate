"""
fenix_core: shared configuration, data loading and pre-processing functions for the FENIX frame-rate study.

One copy of every shared function, imported by both notebooks (01_pipeline, 02_diagnostics), so both run the same code.

Usage (Kaggle)
    %pip install -q "hylite==1.41"
    import sys; sys.path.insert(0, REPO)
    import fenix_core as fx
    fx.setup()                    # provenance, header checks, load data
    from fenix_core import *      # constants, data and functions into the notebook
    (After any later fx.setup() / fx.load(), run `from fenix_core import *` again.)

Refreshing the code without restarting the kernel
    !git -C {REPO} pull -q
    importlib.reload(fx); from fenix_core import *
    Loaded data are kept; only re-run fx.setup() if sections 1–4 (configuration, headers, loading) changed.

Sections
    1. Configuration       runs, paths, output folder, plot styles
    2. Provenance          package versions, Python, platform, run time, git commit of this file
    3. Header checks       fps, integration times, band split; header keys that differ between runs
    4. Load raw data       image, dark, white cubes; band split, detector ceilings, interior columns
    5. Dark frames         bit-8 fix, per-pixel references, dark statistics; dark cuts
    6. White − dark        per-pixel response to the white panel; white − dark cuts
    7. Defect map          dark and white − dark flags per core; fixed defects (flagged in all 4 cores)
    8. Crop                tray interior, the same for all 4 cores
    9. Views               image composites for display (true colour, any band triple)

Refs
    hylite: Thiele et al. 2021, Ore Geol. Rev. 136, 104252, doi:10.1016/j.oregeorev.2021.104252.
    Fischer et al. 2007; Kieffer 1996; EMVA 1288 Release 4.0 General (2021), §8.1.
"""

import datetime
import json
import os
import pathlib
import platform
import subprocess
import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from scipy.ndimage import median_filter


# ----------------------------------------------------------------------------------------------------------------------
# 1. Configuration
# - Four scans of the same core tray (AisaFENIX, LUMO Scanner, 2025-06-03) at increasing frame rates.
#   Labels are folder names, not frame rates.
# - OUT: output folder for all results (env var FENIX_OUT, or /kaggle/working/out).
# - Plot styles: STY (notebook plots); GRSL, STYLE (paper figures: IEEE one column, serif 9 pt, Okabe–Ito colours with
#   distinct markers and line styles).
# ----------------------------------------------------------------------------------------------------------------------

DATA_ROOT = "/kaggle/input/datasets/areeshah/frame-rate-data/Frame Rate Data/Fenix"
RUNS = {"Core 30": (26.46, "Core_30_1_0m00_1m00_2025-06-03_16-06-58"),
        "Core 40": (35.21, "Core_40_1_0m00_1m00_2025-06-03_16-22-34"),
        "Core 50": (43.86, "Core_50_1_0m00_1m00_2025-06-03_16-33-46"),
        "Core 60": (52.63, "Core_60_1_0m00_1m00_2025-06-03_16-44-28")}
CORE_NAMES, FPS = list(RUNS), {n: v[0] for n, v in RUNS.items()}
PATHS = {n: {k: f"{DATA_ROOT}/{run}/capture/{p}{run}.hdr"
             for k, p in [("image", ""), ("dark", "DARKREF_"), ("white", "WHITEREF_")]}
         for n, (_, run) in RUNS.items()}
OUT = globals().get("OUT") or os.environ.get("FENIX_OUT", "/kaggle/working/out")

STY = [("-", "o"), ("--", "s"), ("-.", "^"), (":", "D")]
GRSL = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "lines.linewidth": 1, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600}
STYLE = [("#000000", "o", "-"), ("#E69F00", "s", "--"), ("#56B4E9", "^", "-."), ("#009E73", "D", ":")]

CEIL = {"VNIR": 4095, "SWIR": 65535}             # detector ceilings: VNIR 12-bit, SWIR 16-bit (hylite fenix.py)
INT = slice(1, 383)                              # interior columns; edges x = 0, 383 left out of all statistics

# Crop: tray interior (core pieces, wooden dividers, trough bottoms); tray edges and the empty fifth slot removed.
# - Columns x 35–284: 5 pixels inside the tray walls (x = 30, 290), so no wall pixels are included.
# - Lines 130–609: leaves out the glare on the top wooden strip.
# - The same crop for all 4 cores; the cores are offset from each other by at most 2 lines / 2 columns (registration
#   evidence), within the 5-pixel margin.
XCROP, LCROP = slice(30 + 5, 290 - 5), slice(130, 610)
# Loaded state, set by setup(). Kept across importlib.reload(fenix_core), so refreshing the code does not reload
# the data (see refresh() in the notebooks); lru caches are rebuilt on reload.
HDR, cores, wl, r, ARRAYS, CEIL_B = (globals().get(k) for k in ("HDR", "cores", "wl", "r", "ARRAYS", "CEIL_B"))


def setup(out=None, verbose=True):
    """Provenance, header checks and data loading, in that order."""
    global OUT
    if out:
        OUT = out
    env = provenance()
    if verbose:
        print(json.dumps(env, indent=2))
    check_headers(verbose)
    load(verbose)


# ----------------------------------------------------------------------------------------------------------------------
# 2. Provenance
# - Saves the installed package versions, Python version, platform, run time (UTC) and the git commit of this file.
# - Outputs: OUT/provenance/env.json; OUT/provenance/all_packages.txt (full pip freeze).
# ----------------------------------------------------------------------------------------------------------------------

def _ver(p):
    try:
        return version(p)
    except PackageNotFoundError:
        return None


def _git_commit():
    try:
        out = subprocess.run(["git", "-C", str(pathlib.Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True)
        return out.stdout.strip() or None
    except OSError:
        return None


def provenance():
    prov = pathlib.Path(OUT, "provenance")
    prov.mkdir(parents=True, exist_ok=True)
    env = {p: _ver(p) for p in ("hylite", "numpy", "scipy", "pandas", "scikit-image", "matplotlib",
                                "opencv-python", "opencv-python-headless")}
    env |= {"python": sys.version.split()[0], "platform": platform.platform(),
            "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "fenix_core_commit": _git_commit()}
    (prov / "env.json").write_text(json.dumps(env, indent=2))
    (prov / "all_packages.txt").write_text(
        subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    return env


# ----------------------------------------------------------------------------------------------------------------------
# 3. Header checks
# - fps matches each ENVI header ('fps'; 'fps2' must match).
# - tint1 (VNIR), tint2 (SWIR) and vimg2 (VNIR/SWIR band split) are equal in the image, dark and white of a run.
# - Integration times are equal across runs (within 0.01 %).
# - Header keys that differ between runs are listed, to see what changed besides fps.
#
# Result
# - Integration times: VNIR 17 ms in all runs; SWIR 12.9984 ms (Core 30) vs 12.9992 ms (others), a 0.006 % difference,
#   negligible. SWIR detector temperature 155 in all runs (units not stated).
# - Runs were acquired in frame-rate order (Start Time 16:07 → 16:44): frame rate and acquisition time cannot be
#   separated.
# - Besides fps, only these header values differ: Start/Stop Time, lines (735–739), tint2 (above), fps_qpf (meaning
#   unknown; within 0.6 % of fps) and the Scb temperatures (≤ 0.4 apart; units not stated).
# ----------------------------------------------------------------------------------------------------------------------

def header(p):
    """Single-line 'key = value' entries of an ENVI header."""
    with open(p) as f:
        return {k.strip(): v.strip() for k, v in (line.split("=", 1) for line in f if "=" in line)}


def check_headers(verbose=True):
    global HDR
    missing = [p for d in PATHS.values() for p in d.values() if not os.path.exists(p)]
    assert not missing, f"missing files: {missing}"

    HDR = {n: {k: header(p) for k, p in PATHS[n].items()} for n in CORE_NAMES}
    for n in CORE_NAMES:
        for k, h in HDR[n].items():
            assert float(h["fps"]) == FPS[n] and h["fps2"] == h["fps"], f"{n} {k}: fps {h['fps']}, fps2 {h['fps2']}"
        for t in ("tint1", "tint2", "vimg2"):
            assert len({HDR[n][k][t] for k in HDR[n]}) == 1, f"{n}: {t} differs between image, dark and white"
    for t in ("tint1", "tint2"):
        v = np.array([float(HDR[n]["image"][t]) for n in CORE_NAMES])
        assert np.ptp(v) <= 1e-4 * v.mean(), f"{t} differs between runs: {v}"

    if verbose:
        print(f"{'core':8} {'fps':>6} {'tint1 (ms)':>10} {'tint2 (ms)':>10} {'SWIR T':>7}  start (UTC)")
        for n in CORE_NAMES:
            h = HDR[n]["image"]
            print(f"{n:8} {FPS[n]:>6} {float(h['tint1']):>10.4f} {float(h['tint2']):>10.4f} {h['SWIR temperature']:>7}  "
                  f"{h['Start Time'].split(':', 1)[1].strip()}")
        keys = set.intersection(*(set(HDR[n]["image"]) for n in CORE_NAMES))
        diff = sorted(k for k in keys if len({HDR[n]["image"][k] for n in CORE_NAMES}) > 1)
        print("\nHeader keys that differ between runs (image headers):")
        for k in diff:
            print(f"  {k}: " + " | ".join(HDR[n]["image"][k][:30] for n in CORE_NAMES))


# ----------------------------------------------------------------------------------------------------------------------
# 4. Load raw data
# - Loads the raw DN image, dark and white reference cubes of each scan (hylite io.load).
# - Band split from the header (vimg2 = {175, 448}: first SWIR band 175, 1-based → r = 174); CEIL_B: ceiling per band.
# - Raw FENIX axes: [x (384 cross-track samples), line (frame), band] (fenix.py: xdim() == 384).
#
# Checks
# - Every cube (image, dark, white) has 384 cross-track samples and the same wavelengths.
# - No reading is above its detector ceiling (confirms CEIL, used by all later saturation tests).
# - Image cubes are raw DN (maximum > 1000).
#
# Result
# - 448 bands, 377.79–2502.15 nm: VNIR 174 bands (377.79–971.13 nm), SWIR 274 bands (978.09–2502.15 nm).
# - Lines: 739, 737, 735, 736. Dark: 50 frames per scan. White: 18, 24, 29, 35 frames (rising with fps).
# ----------------------------------------------------------------------------------------------------------------------

def load(verbose=True):
    global cores, wl, r, ARRAYS, CEIL_B
    from hylite import io

    cores = {n: {k: io.load(p) for k, p in PATHS[n].items()} for n in CORE_NAMES}
    wl = np.asarray(cores[CORE_NAMES[0]]["image"].get_wavelengths())
    r = int(HDR[CORE_NAMES[0]]["image"]["vimg2"].strip("{} ").split(",")[0]) - 1     # first SWIR band (0-based)
    ARRAYS = {"VNIR": slice(0, r), "SWIR": slice(r, None)}
    CEIL_B = np.where(np.arange(wl.size) < r, CEIL["VNIR"], CEIL["SWIR"])
    dark.cache_clear(); ref.cache_clear(); defect_map.cache_clear()                  # new data: drop cached results

    if verbose:
        print(f"{'core':8} {'fps':>6} | {'lines':>5} {'dark':>4} {'white':>5} | max DN VNIR, SWIR (image)")
    for n in CORE_NAMES:
        for k, c in cores[n].items():
            assert c.xdim() == 384, f"{n} {k}: {c.xdim()} cross-track samples"
            assert np.allclose(c.get_wavelengths(), wl), f"{n} {k}: wavelength mismatch"
            for det, sl in ARRAYS.items():
                assert c.data[:, :, sl].max() <= CEIL[det], f"{n} {k} {det}: reading above {CEIL[det]} DN"
        img = cores[n]["image"].data
        assert img.max() > 1000, f"{n}: image is not raw DN"
        if verbose:
            print(f"{n:8} {FPS[n]:>6} | {img.shape[1]:>5} {cores[n]['dark'].ydim():>4} {cores[n]['white'].ydim():>5} | "
                  f"{img[:, :, ARRAYS['VNIR']].max()}, {img[:, :, ARRAYS['SWIR']].max()}")
    if verbose:
        print(f"\nbands = {wl.size} ({wl[0]:.2f}–{wl[-1]:.2f} nm) | VNIR {r} bands ({wl[0]:.2f}–{wl[r - 1]:.2f} nm) | "
              f"SWIR {wl.size - r} bands ({wl[r]:.2f}–{wl[-1]:.2f} nm)")


# ----------------------------------------------------------------------------------------------------------------------
# 5. Dark frames: bit-8 fix, per-pixel references and dark statistics
# - bit8_fix: undoes bit-8 errors in repeated frames. A reading 128–384 DN from its pixel's median over the frames is
#   moved back by 256 DN, only in the direction its own bit 8 allows (bit 8 = 1: can only be too high; 0: too low).
#   Readings further off are left as they are (not one bit error).
# - dark(n, det): dark frames (x, frame, band); SWIR bit-8 corrected, VNIR raw (no bit-8 errors).
# - ref(n, kind, det): per-pixel reference (x, band). Dark: mean of dark(). White: VNIR mean, SWIR median of the raw
#   frames (white noise, ≈ 70–100 DN, is too close to the 128 DN window to correct safely; the median is barely
#   affected by the jumps).
# - dark() and ref() are cached and read-only: each is computed once per (core, detector), and cannot be changed in
#   place by later code.
# - dark_stats(n, det): per pixel, from the 50 dark frames: mean − band median (DN) → hot / cold;
#   std / band-median std → noisy / stuck; any frame at the detector ceiling → sat. Baselines from interior columns.
#
# Constants (from the dark threshold evidence, diagnostics notebook)
# - CUTS: SWIR cuts on mean and noise, each inside a gap found in all 4 cores. VNIR: no gaps, so no cuts.
# - K_FRAME = 6: a frame jump counts if > 6 × band noise; normal noise reaches it less than once per core.
# - NOISE_CUT = 2.5: only pixels with robust noise ≤ 2.5 × band noise are tested for flashers (our choice; no gap).
#
# Result (diagnostics notebook)
# - Bit-8 fix: 10.05–10.18 % of SWIR dark readings corrected; 0.004–0.007 % still > 128 DN from the pixel median;
#   pixel noise 21.5 DN (median), far below the 128 DN window.
# - Dark reference: a raw-median SWIR dark would add a 0.04 % cross-track stripe (dark-reference check), so the
#   bit-8 corrected mean is used.
#
# Refs
# - Fischer et al. 2007 (dark mean, noise and flasher tests); Kieffer 1996 (band-normalized statistics, natural gaps).
# ----------------------------------------------------------------------------------------------------------------------

CUTS = {"mean": (-6000, 7500), "noise": (0.07, 10)}             # SWIR (low, high); VNIR: no cuts
K_FRAME, NOISE_CUT = 6, 2.5


def bit8_fix(a):
    """Undo bit-8 errors in repeated frames a (x, frame, band). Returns corrected frames and the correction mask."""
    dv = a - np.median(a, 1, keepdims=True)
    b8 = (a.astype(np.int64) >> 8) & 1
    down = (b8 == 1) & (dv > 128) & (dv < 384)
    up = (b8 == 0) & (dv < -128) & (dv > -384)
    return a - 256 * down + 256 * up, down | up


def _readonly(a):
    a.setflags(write=False)
    return a


@lru_cache(maxsize=None)
def dark(n, det):
    """Dark frames (x, frame, band): SWIR bit-8 corrected, VNIR raw. Cached, read-only."""
    a = cores[n]["dark"].data[:, :, ARRAYS[det]].astype(np.float64)
    return _readonly(bit8_fix(a)[0] if det == "SWIR" else a)


@lru_cache(maxsize=None)
def ref(n, kind, det):
    """Per-pixel reference (x, band): dark = mean of dark(); white = VNIR mean, SWIR median of raw frames. Cached."""
    if kind == "dark":
        return _readonly(dark(n, det).mean(1))
    A = cores[n]["white"].data[:, :, ARRAYS[det]].astype(np.float64)
    return _readonly(A.mean(1) if det == "VNIR" else np.median(A, 1))


def dark_stats(n, det):
    """Per-pixel dark statistics (x, band); band baselines from the interior columns (INT)."""
    a = dark(n, det)
    mu, s = a.mean(1), a.std(1, ddof=1)
    return {"mean": mu - np.median(mu[INT], 0), "noise": s / np.median(s[INT], 0),
            "sat": np.any(cores[n]["dark"].data[:, :, ARRAYS[det]] >= CEIL[det], 1)}


# ----------------------------------------------------------------------------------------------------------------------
# 6. White − dark: per-pixel response to the white panel
# - wd_mean(n, det): white − dark per pixel (x, band) from the references (ref), divided by its band's median over the
#   interior columns, so a normal pixel = 1.0; plus a mask of pixels at the detector ceiling in any white or dark frame
#   (clipped: response not measurable). Kieffer 1996: a uniform source "minus the dark level and is band-normalized it
#   represents the raw responsivity of the detector".
# - prnu_dev(n, det): % deviation of each pixel from the median of its HP_K neighbouring columns (itself and 2 on each
#   side) in the same band; clipped pixels NaN. Removes the uneven lighting of the white panel across the track, so
#   only the pixels themselves are tested. EMVA 1288 §8.1 applies a highpass filter "to show the properties of the
#   camera rather than the properties of an imperfect illumination system"; EMVA uses box filters, the column median
#   is our adaptation.
#
# Constants (from the white − dark threshold evidence, diagnostics notebook)
# - WCUTS: % deviation (low, high), each inside a gap found in all 4 cores. SWIR −25 / +7 %; VNIR −20 % (no gap common
#   to all 4 cores on the high side, so no high cut).
# - HP_K = 5: the smallest window that flags the same pixels as all wider windows tested (7, 9, 11).
#
# Refs
# - Kieffer 1996; EMVA 1288 Release 4.0 General (2021), §8.1.
# ----------------------------------------------------------------------------------------------------------------------

WCUTS = {"SWIR": (-25, 7), "VNIR": (-20, np.inf)}               # % deviation (low, high)
HP_K = 5


def wd_mean(n, det):
    """Band-normalized white − dark (x, band), and mask of pixels clipped in any white or dark frame."""
    sl = ARRAYS[det]
    P = ref(n, "white", det) - ref(n, "dark", det)
    sat = (cores[n]["white"].data[:, :, sl] >= CEIL[det]).any(1) | (cores[n]["dark"].data[:, :, sl] >= CEIL[det]).any(1)
    return P / np.median(P[INT], 0), sat


def prnu_dev(n, det, k=HP_K):
    """% deviation (x, band) of each pixel from the median of k neighbouring columns (same band); clipped pixels NaN."""
    Pn, sat = wd_mean(n, det)
    dev = 100 * (Pn - median_filter(Pn, size=(k, 1), mode="nearest"))
    dev[sat] = np.nan
    return dev


# ----------------------------------------------------------------------------------------------------------------------
# 7. Defect map: fixed defects from the dark and white − dark tests
# - dark_flags(n): pixels {(x, band)} per class from the dark frames. SWIR: all tests; VNIR: saturation only (no gaps,
#   so no mean or noise cuts; no flashers; dark threshold evidence). Classes (Fischer et al. 2007):
#     hot / cold    : dark level above / below CUTS["mean"].
#     noisy / stuck : noise above / below CUTS["noise"] ("too much variation" / "little or no variation").
#     sat           : at the detector ceiling in any dark frame ("a signal fixed at saturation").
#     flasher       : a quiet pixel (robust noise ≤ NOISE_CUT) with jumps > K_FRAME × band noise in ≥ 2 frames
#                     ("intermittently bad across several frames").
# - white_flags(n): pixels per class from the white − dark response (prnu_dev), at most one class per pixel
#   (Kieffer 1996):
#     low  : below the low cut ("very low response or even 'dead' pixels are clearly to be excluded").
#     over : above the high cut, SWIR only ("exceedingly responsive pixels as suspect").
#     sat  : at the detector ceiling in any white or dark frame; response not measurable.
# - defect_map(): fixed defects = flagged in all 4 cores, by either test. Only these are masked (later), so the mask is
#   the same at every frame rate. Run-specific flags (some cores only) stay in the data, so any frame-rate effect they
#   carry stays in the noise. Cached; returns read-only views (frozensets, tuples).
# - defect_mask(det): fixed defects as a boolean array (x, band) over the detector's bands.
# - Every class is masked, including those that only shift a pixel's offset or gain: such a shift cancels in
#   reflectance only if the pixel responds linearly, which one white level cannot test (Kieffer 1996: "It is critical to
#   understand which elements are suspect to avoid misleading results").
#
# Result
# - Dark, fixed: 33 SWIR pixels (hot 24, cold 9, noisy 1, stuck 2, sat 2; a pixel can be in several classes); no VNIR.
#   Dark, run-specific (kept): 54 SWIR pixels, almost all flashers, 20 / 25 / 18 / 11 per core (26.46 → 52.63 fps),
#   not in frame-rate order; 41 flagged in one core, 6 in two, 7 in three; plus 1 stuck pixel in Core 60.
# - White − dark, fixed: 44 pixels (low 14, over 11, sat 19); no run-specific flags.
# - Defect map: 49 pixels (47 SWIR, 2 VNIR) = white − dark 44 + dark-only 5; 28 flagged by both tests. Most saturated
#   pixels are also hot in the dark; every weak SWIR pixel is also bad in the dark; over-responsive pixels sit next to
#   weak ones (e.g. x = 136, band 377; Kieffer 1996: "The most responsive (band-normalized) pixels are adjacent to weak
#   or non-responsive pixels").
# - Frame rate: no effect on fixed defects; the map is the same for all 4 cores.
#
# Refs
# - Fischer et al. 2007; Kieffer 1996.
# ----------------------------------------------------------------------------------------------------------------------

DARK_CLASSES = ("hot", "cold", "noisy", "stuck", "sat", "flasher")
WHITE_CLASSES = ("low", "over", "sat")


def _pixels(mask, sl):
    """(x, band) pairs of a boolean (interior x, detector band) mask, in full-cube indices."""
    return {(int(x) + INT.start, int(b) + sl.start) for x, b in np.argwhere(mask)}


def dark_flags(n):
    """Flagged pixels {(x, band)} per dark class for core n, interior columns only."""
    out = {k: set() for k in DARK_CLASSES}
    for det, sl in ARRAYS.items():
        st = {k: v[INT] for k, v in dark_stats(n, det).items()}
        tests = {"sat": st["sat"]}
        if det == "SWIR":
            a = dark(n, det)[INT]
            d, ms = np.abs(a - np.median(a, 1, keepdims=True)), np.median(a.std(1, ddof=1), 0)
            quiet = 1.4826 * np.median(d, 1) / ms <= NOISE_CUT
            tests |= {"hot": st["mean"] > CUTS["mean"][1], "cold": st["mean"] < CUTS["mean"][0],
                      "noisy": st["noise"] > CUTS["noise"][1], "stuck": st["noise"] < CUTS["noise"][0],
                      "flasher": quiet & ((d > K_FRAME * ms).sum(1) >= 2)}
        for k, m in tests.items():
            out[k] |= _pixels(m, sl)
    return out


def white_flags(n):
    """Flagged pixels {(x, band)} per white − dark class for core n, interior columns only."""
    out = {k: set() for k in WHITE_CLASSES}
    for det, sl in ARRAYS.items():
        dev = prnu_dev(n, det)[INT]                                       # clipped pixels NaN
        lo, hi = WCUTS[det]
        for k, m in (("low", dev <= lo), ("over", dev >= hi), ("sat", np.isnan(dev))):
            out[k] |= _pixels(m, sl)
    return out


@lru_cache(maxsize=None)
def defect_map():
    """Fixed defects (flagged in all 4 cores) and the per-core flags behind them. Cached; read-only contents."""
    DK = {n: {k: frozenset(v) for k, v in dark_flags(n).items()} for n in CORE_NAMES}
    WK = {n: {k: frozenset(v) for k, v in white_flags(n).items()} for n in CORE_NAMES}
    dfix = {k: frozenset.intersection(*(DK[n][k] for n in CORE_NAMES)) for k in DARK_CLASSES}
    wfix = {k: frozenset.intersection(*(WK[n][k] for n in CORE_NAMES)) for k in WHITE_CLASSES}
    pixels = tuple(sorted(frozenset().union(*dfix.values(), *wfix.values())))
    assert all(INT.start <= x < INT.stop for x, _ in pixels), "defect at an edge column"
    classes = {c: tuple([f"dark:{k}" for k in DARK_CLASSES if c in dfix[k]] +
                        [f"white:{k}" for k in WHITE_CLASSES if c in wfix[k]]) for c in pixels}
    return {"pixels": pixels, "classes": classes, "dark_fixed": dfix, "white_fixed": wfix,
            "dark_flags": DK, "white_flags": WK}


def defect_mask(det):
    """Fixed defects as a boolean array (x, band) over the bands of detector det."""
    b0, b1, _ = ARRAYS[det].indices(wl.size)
    m = np.zeros((cores[CORE_NAMES[0]]["dark"].data.shape[0], b1 - b0), bool)
    for x, b in defect_map()["pixels"]:
        if b0 <= b < b1:
            m[x, b - b0] = True
    return m


# ----------------------------------------------------------------------------------------------------------------------
# 8. Crop: tray interior (core pieces, wooden dividers, trough bottoms), the same for all 4 cores
# - Columns x 35–284: 5 pixels inside the tray walls (x = 30, 290), so no wall pixels are included; the empty fifth
#   slot is removed.
# - Lines 130–609: leaves out the glare on the top wooden strip.
# - Chosen by visual inspection of the raw true-colour view (diagnostics D8). The cores are offset from each other by
#   at most 2 lines / 2 columns (registration, clustering notebook), within the 5-pixel margin.
# - A fixed defect (x, band) affects every line of its column, so whether it lies inside the crop depends only on x.
#
# Result
# - Each core: (384, 735–739, 448) → (250, 480, 448). 33 of the 49 fixed defects lie inside the crop (31 SWIR, 2 VNIR).
# ----------------------------------------------------------------------------------------------------------------------

LINE_START, LINE_END = 130, 610
COL_START, COL_END = 30 + 5, 290 - 5
CROP_X, CROP_L = slice(COL_START, COL_END), slice(LINE_START, LINE_END)


# ----------------------------------------------------------------------------------------------------------------------
# 9. Views: image composites for display only (nothing here enters any computation)
# - composite(A, wl_arr, nm): bands nearest to the wavelengths nm (one band → greyscale, three → colour), each channel
#   stretched to its own 2–98 % range; NaN shown white. A: (x, line, band) → image (line, x, len(nm)) in [0, 1].
# - true_rgb(A, wl_arr): true colour, 680 / 550 / 505 nm (RGB_NM; hylite's RGB preset).
# ----------------------------------------------------------------------------------------------------------------------

RGB_NM = (680.0, 550.0, 505.0)


def composite(A, wl_arr, nm, lo=2, hi=98):
    """A: (x, line, band) -> (line, x, len(nm)) image in [0, 1]; per-channel 2–98 % stretch; NaN shown white."""
    c = A[..., [int(np.argmin(np.abs(wl_arr - w))) for w in nm]].astype(np.float32)
    vmin, vmax = np.nanpercentile(c, [lo, hi], axis=(0, 1))
    return np.transpose(np.nan_to_num(np.clip((c - vmin) / (vmax - vmin + 1e-9), 0, 1), nan=1.0), (1, 0, 2))


def true_rgb(A, wl_arr):
    """True colour (RGB_NM) of A (x, line, band) -> (line, x, 3) image in [0, 1]."""
    return composite(A, wl_arr, RGB_NM)