# panel_reflectance_pipeline.py
"""
Pipeline for automated VNIR + SWIR panel detection from hyperspectral cubes.
Includes:
- Datacube import with orientation correction
- VNIR panel detection (seed + region-growing)
- SWIR local search panel detection guided by VNIR
- Visualization helpers for VNIR/SWIR overlay

Usage:
    import panel_reflectance_pipeline as adp
    adp.set_config(vnir_min_patch=8, swir_max_candidates=3000)
    bbox, shape, png, jsonp = adp.run_vnir_detection(path_to_vnir_bin)
    ...
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from collections import deque
from skimage import measure, morphology
import json

# ============================================================
# 1 Datacube import with orientation + preview
# ============================================================
# We need to be able to see what we are doing, to find the box
# (reflectance panel) which is quite the task, especially in the SWIR wavelengths!

def import_datacube(bin_path, verify_orientation=True, save_preview=True, dtype=np.uint16):
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f"Datacube not found: {bin_path}")

    # parse dims (expects digits in filename)
    parts = bin_path.stem.split("_")
    nums = [int(p) for p in parts if p.isdigit()]
    if len(nums) < 3:
        raise ValueError(f"Cannot infer dimensions from filename: {bin_path.name}")

    bands, frames, spatial = nums[:3]

    # detect sensor type heuristics
    is_vnir = "reorder_272" in bin_path.stem.lower() or "hyper" in str(bin_path.parent).lower()
    is_swir = "reorder_288" in bin_path.stem.lower() or "swir" in str(bin_path.parent).lower()

    # load and reshape -> (bands, frames, spatial) then transpose to (Y, X, Bands)
    cube = np.fromfile(bin_path, dtype=dtype)
    expected = bands * frames * spatial
    if cube.size != expected:
        raise ValueError(f"Expected {expected} values but got {cube.size} from {bin_path}")

    cube = cube.reshape((bands, frames, spatial))
    cube = np.transpose(cube, (2, 1, 0))  # (Y, X, Bands)

    # correct orientation for SWIR (flipud)
    if is_swir:
        cube = np.flipud(cube)

    # assign wavelengths (simple linspace)
    if is_vnir:
        wavelengths = np.linspace(400, 1000, bands)
    elif is_swir:
        wavelengths = np.linspace(1000, 2500, bands)
    else:
        wavelengths = np.linspace(400, 2500, bands)

    # preview optionally
    if verify_orientation:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(top=0.75)
        if is_vnir:
            rgb_wvls = [465, 514, 656]
            rgb_idx = [np.argmin(np.abs(wavelengths - w)) for w in rgb_wvls]
            rgb = cube[:, :, rgb_idx].astype(np.float32)
            rgb = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min()), 0, 1)
            ax.imshow(rgb)
            ax.set_title(f"VNIR RGB composite ({bin_path.name})")
        else:
            wl_target = 1550
            idx = np.argmin(np.abs(wavelengths - wl_target))
            gray = cube[:, :, idx]
            ax.imshow(gray, cmap="gray")
            ax.set_title(f"SWIR {wl_target:.0f} nm grayscale ({bin_path.name})")
        ax.axis("off")

        Y, X, Z = cube.shape
        fig.text(0.1, 0.92, f"Y-axis: {Y}", fontsize=10)
        fig.text(0.1, 0.88, f"X-axis: {X}", fontsize=10)
        fig.text(0.1, 0.84, f"Z-axis: {Z}", fontsize=10)
        fig.text(0.7, 0.92, "X →", color='red', fontsize=12)
        fig.text(0.9, 0.92, "Y ↑", color='red', fontsize=12)
        fig.text(0.85, 0.88, "Z (bands)", color='red', fontsize=12)

        if save_preview:
            out_png = bin_path.parent / f"{bin_path.stem}_preview.png"
            fig.savefig(out_png, dpi=200)
            print(f"Saved preview → {out_png.name}")

        plt.show()

    return cube.astype(np.float32), wavelengths


# ============================================================
# 2 Visualization helpers
# ============================================================

def visualize_panel_detection(cube, wavelengths, panel_bbox, save_path=None, sensor_type="VNIR"):
    fig, ax = plt.subplots(figsize=(10, 5))
    x1, y1, x2, y2 = panel_bbox
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')

    if sensor_type == "VNIR":
        rgb_wvls = [465, 514, 656]
        rgb_idx = [np.argmin(np.abs(wavelengths - w)) for w in rgb_wvls]
        rgb = cube[:, :, rgb_idx].astype(np.float32)
        rgb = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min()), 0, 1)
        ax.imshow(rgb)
        ax.set_title("VNIR RGB with Panel Detection")
    else:
        wl_target = 1550
        idx = np.argmin(np.abs(wavelengths - wl_target))
        gray = cube[:, :, idx]
        ax.imshow(gray, cmap="gray")
        ax.set_title(f"SWIR {wl_target:.0f} nm with Panel Detection")

    ax.add_patch(rect)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved panel overlay → {Path(save_path).name}")
    plt.show()


# ============================================================
# 3 Other helpers - panel detection
# ============================================================
# These helpers assist us on our quest to find the box (reflectance panel)
# in both VNIR and SWIR datacubes.

def select_informative_swir_wavelengths(wavelengths):
    target_bands = [1000, 1300, 1550, 1800, 2200]
    return [np.argmin(np.abs(wavelengths - w)) for w in target_bands]

def robust_adaptive_threshold(region_sm, percentile=95, std_factor=1.5):
    p_thresh = np.nanpercentile(region_sm, percentile)
    s = np.nanstd(region_sm)
    return p_thresh + std_factor * s

def ensure_python_numbers(obj):
    if isinstance(obj, dict):
        return {k: ensure_python_numbers(v) for k,v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        cls = type(obj)
        return cls(ensure_python_numbers(x) for x in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj

def normalize_for_detection(region):
    p1, p99 = np.nanpercentile(region, [1, 99])
    if p99 - p1 < 1e-6:
        return np.zeros_like(region)
    return np.clip((region - p1) / (p99 - p1), 0, 1)

def advanced_morphological_cleanup(mask, min_area=10, max_area_ratio=0.3):
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_area)
    labeled = measure.label(mask)
    props = measure.regionprops(labeled)
    total_area = mask.shape[0] * mask.shape[1]
    filtered_mask = np.zeros_like(mask)
    for prop in props:
        if prop.area <= max_area_ratio * total_area:
            filtered_mask[labeled==prop.label] = True
    filtered_mask = morphology.binary_closing(filtered_mask, morphology.square(5))
    filtered_mask = morphology.binary_opening(filtered_mask, morphology.square(3))
    return filtered_mask

def _clip_bbox(bbox, h, w):
    x1, y1, x2, y2 = bbox
    x1 = int(np.clip(int(x1), 0, w-1)); x2 = int(np.clip(int(x2), 0, w-1))
    y1 = int(np.clip(int(y1), 0, h-1)); y2 = int(np.clip(int(y2), 0, h-1))
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    return (x1, y1, x2, y2)

def map_vnir_to_swir_direct(vnir_bbox, vnir_shape, swir_shape):
    x1, y1, x2, y2 = map(int, vnir_bbox)
    v_h, v_w = int(vnir_shape[0]), int(vnir_shape[1])
    s_h, s_w = int(swir_shape[0]), int(swir_shape[1])
    x1_s = int(round(x1 * s_w / v_w))
    x2_s = int(round(x2 * s_w / v_w))
    y1_s = int(round(y1 * s_h / v_h))
    y2_s = int(round(y2 * s_h / v_h))
    return _clip_bbox((x1_s, y1_s, x2_s, y2_s), s_h, s_w)

def build_expanded_search_box(swir_bbox, swir_shape, pct_x=0.05, pct_y=0.2):
    s_h, s_w = swir_shape[:2]
    x1, y1, x2, y2 = swir_bbox
    pad_x = int(round(pct_x * s_w))
    pad_y = int(round(pct_y * s_h))
    return _clip_bbox((x1-pad_x, y1-pad_y, x2+pad_x, y2+pad_y), s_h, s_w)

## test - remove these helpers as well:
# Helper: extract panel subcube
def extract_panel_subcube(full_cube, detected_bbox_full):
    x1, y1, x2, y2 = map(int, detected_bbox_full)
    return full_cube[y1:y2+1, x1:x2+1, :]

# Helper: visualize full-cube SWIR overlay
def visualize_swir_panel_overlay(cube, bbox, wavelengths, save_path):
    """
    Visualize SWIR full-cube overlay using a single wavelength (1550nm) in greyscale,
    with detected panel bbox overlaid.
    """
    # --- Find the band index closest to 1550nm ---
    target_wavelength = 1550
    band_idx = int(np.argmin(np.abs(wavelengths - target_wavelength)))

    # --- Extract that band ---
    img = cube[:, :, band_idx].astype(np.float32)
    mn, mx = np.nanmin(img), np.nanmax(img)
    if mx - mn < 1e-6:
        img_norm = np.zeros_like(img)
    else:
        img_norm = (img - mn) / (mx - mn)

    # --- Plot greyscale ---
    x1, y1, x2, y2 = map(int, bbox)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(img_norm, cmap='gray')
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                           edgecolor='lime', facecolor='none', linewidth=2))
    ax.set_title(f"SWIR full-cube overlay ({target_wavelength}nm)")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"✅ Full SWIR overlay saved (greyscale {target_wavelength}nm): {save_path}")


# ============================================================
# 4 Other helpers - reflectance conversion
# ============================================================
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def read_detection_json(json_path: Path):
    """Read VNIR or SWIR detection JSON and return .bin path + bbox."""
    with open(json_path, "r") as f:
        data = json.load(f)
    bin_path = Path(data["file"]) if "file" in data else None
    bbox = None
    source = "unknown"
    if "bbox" in data and "vnir" in json_path.name.lower():
        bbox = tuple(map(int, data["bbox"]))
        source = "VNIR"
    elif "detected_bbox" in data:
        bbox = tuple(map(int, data["detected_bbox"]))
        source = "SWIR"
    elif "mapped_direct_bbox" in data:
        bbox = tuple(map(int, data["mapped_direct_bbox"]))
        source = "SWIR-mapped"
    else:
        raise KeyError(f"No usable bbox found in JSON: {json_path}")
    return bin_path, bbox, data, source

def extract_panel_region(cube, bbox):
    """Extract panel subcube using bounding box."""
    x1, y1, x2, y2 = map(int, bbox)
    return cube[y1:y2+1, x1:x2+1, :]

def compute_calibration_coeff(panel_cube, panel_cal):
    """Compute per-band calibration coefficient: counts / known reflectance."""
    panel_median = np.nanmedian(panel_cube, axis=(0,1))
    num_bands = len(panel_median)
    cal_interp = np.interp(np.arange(num_bands), panel_cal["wavelength"], panel_cal["value"])
    cal_coeff = panel_median / cal_interp  # DN / known reflectance
    return cal_coeff

def normalize_cube_to_panel(cube, cal_coeff, epsilon=1e-6):
    """Convert raw cube to reflectance using calibration coefficient."""
    safe_coeff = np.where(cal_coeff == 0, epsilon, cal_coeff)
    return cube / safe_coeff  # broadcast along bands

def export_reflectance_netcdf(reflectance_cube, wavelengths, save_path):
    """Save reflectance datacube as NetCDF using netcdf4 engine."""
    ds = xr.Dataset(
        {"reflectance": (("y", "x", "wavelength"), reflectance_cube.astype(np.float32))},
        coords={
            "y": np.arange(reflectance_cube.shape[0]),
            "x": np.arange(reflectance_cube.shape[1]),
            "wavelength": wavelengths
        }
    )
    ds.to_netcdf(save_path, engine='netcdf4')
    print(f"✅ Saved reflectance cube: {save_path.name}")

def quick_reflectance_plot(wavelengths, panel_refl, cube_type, save_csv_path=None):
    """Plot median panel reflectance spectrum and optionally save CSV."""
    plt.figure(figsize=(6, 3))
    plt.plot(wavelengths, panel_refl, lw=1.2)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Median Reflectance")
    plt.title(f"{cube_type} - Median Panel Reflectance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    if save_csv_path:
        df = pd.DataFrame({"wavelength_nm": wavelengths, "median_reflectance": panel_refl})
        df.to_csv(save_csv_path, index=False)
        print(f"✅ Saved median panel reflectance CSV: {save_csv_path.name}")
