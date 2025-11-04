"""
panel_reflectance_pipeline.py
-----------------------------
Enhanced module for VNIR/SWIR spectral datacube reflectance processing.

Updates:
 - Corrected datacube orientation (Y, X, Bands)
 - Automatically generates RGB/Gray previews
 - Automatically detects and overlays panel region
 - Saves panel overlay previews for VNIR/SWIR cubes
"""

import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# 1️⃣ Import Datacube with Orientation + Preview
# ============================================================
def import_datacube(bin_path, verify_orientation=True, save_preview=True, dtype=np.uint16):
    """
    Import a reordered hyperspectral .bin file (3D datacube).
    Detects VNIR vs SWIR automatically, corrects orientation,
    and shows an RGB or grayscale preview for verification.
    For SWIR, flips the Y-axis so origin is bottom-left.
    """
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f"Datacube not found: {bin_path}")

    # Parse dimensions from filename (e.g., reorder_272_2295_640_U16.bin)
    parts = bin_path.stem.split("_")
    nums = [int(p) for p in parts if p.isdigit()]
    if len(nums) < 3:
        raise ValueError(f"Cannot infer dimensions from filename: {bin_path.name}")

    bands, frames, spatial = nums[:3]

    # Identify VNIR vs SWIR
    is_vnir = "reorder_272" in bin_path.stem.lower() or "hyper" in str(bin_path.parent).lower()
    is_swir = "reorder_288" in bin_path.stem.lower() or "swir" in str(bin_path.parent).lower()

    # Load binary and reshape to (bands, frames, spatial)
    cube = np.fromfile(bin_path, dtype=dtype)
    expected = bands * frames * spatial
    if cube.size != expected:
        raise ValueError(f"Expected {expected} values but got {cube.size}")

    cube = cube.reshape((bands, frames, spatial))
    cube = np.transpose(cube, (2, 1, 0))  # → (Y, X, Bands)

    # --- Correct Y-axis orientation for SWIR ---
    if is_swir:
        cube = np.flipud(cube)

    print(f"Loaded cube: {bin_path.name}  shape={cube.shape} (Y={cube.shape[0]}, X={cube.shape[1]}, Bands={cube.shape[2]})")

    # Assign wavelengths
    if is_vnir:
        wavelengths = np.linspace(400, 1000, bands)
    elif is_swir:
        wavelengths = np.linspace(1000, 2500, bands)
    else:
        wavelengths = np.linspace(400, 2500, bands)

    # --- Preview with axes labels and arrows ---
    if verify_orientation:
        fig, ax = plt.subplots(figsize=(10, 5))

        # Add extra top space for annotations
        fig.subplots_adjust(top=0.75)  # reserve top 25% of figure for labels/arrows

        # VNIR RGB
        if is_vnir:
            rgb_wvls = [465, 514, 656]
            rgb_idx = [np.argmin(np.abs(wavelengths - w)) for w in rgb_wvls]
            rgb = cube[:, :, rgb_idx].astype(np.float32)
            rgb = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min()), 0, 1)
            ax.imshow(rgb)
            ax.set_title(f"VNIR RGB composite ({bin_path.name})")

        # SWIR grayscale
        elif is_swir:
            wl_target = 1550
            idx = np.argmin(np.abs(wavelengths - wl_target))
            gray = cube[:, :, idx]
            ax.imshow(gray, cmap="gray")
            ax.set_title(f"SWIR {wl_target:.0f} nm grayscale ({bin_path.name})")

        ax.axis("off")

        # --- Annotate axes with dimensions above image ---
        Y, X, Z = cube.shape
        fig.text(0.1, 0.92, f"Y-axis (spatial pixels): {Y}", color='black', fontsize=10)
        fig.text(0.1, 0.88, f"X-axis (frames/scan): {X}", color='black', fontsize=10)
        fig.text(0.1, 0.84, f"Z-axis (bands/wavelengths): {Z}", color='black', fontsize=10)

        # --- Draw arrows above image ---
        fig.text(0.7, 0.92, "X →", color='red', fontsize=12)
        fig.text(0.9, 0.92, "Y ↑", color='red', fontsize=12)  # arrow now points up
        fig.text(0.85, 0.88, "Z (bands)", color='red', fontsize=12)

        if save_preview:
            out_png = bin_path.parent / f"{bin_path.stem}_preview.png"
            fig.savefig(out_png, dpi=200)
            print(f"Saved preview → {out_png.name}")

        plt.show()

    return cube.astype(np.float32), wavelengths


# ============================================================
# 2️⃣ Detect Panel Region (with auto VNIR→SWIR scaling)
# ============================================================

import numpy as np
import re
from pathlib import Path

def parse_shape_from_filename(filename):
    """
    Parse (bands, width, height) from filenames like:
      reorder_272_2295_640_U16.bin
      Reorder_288_2334_384_U16.bin

    Returns (height, width, bands) as integers.
    """
    filename = Path(filename)
    match = re.search(r"reorder_(\d+)_(\d+)_(\d+)_", filename.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse shape from filename: {filename.name}")
    bands, width, height = map(int, match.groups())
    return height, width, bands


def detect_panel_region(cube, vnird_bbox=None, vnir_file=None, swir_file=None, band_index=None, threshold=None):
    """
    Detect bright calibration panel region in VNIR or SWIR datacubes.

    VNIR:
        - Automatically detects bright region (>90% reflectance).
    SWIR:
        - Converts VNIR bbox → SWIR coordinates using filename-derived shapes.
        - Expands ±15% around scaled bbox, searches for brightest region.
        - Returns a square bounding box centered on bright centroid.

    Parameters
    ----------
    cube : np.ndarray
        Hyperspectral datacube (Y, X, B).
    vnird_bbox : tuple, optional
        Bounding box (x1, y1, x2, y2) from VNIR.
    vnir_file, swir_file : Path-like, optional
        Required for SWIR detection (auto parses cube shape from filename).
    band_index : int, optional
        VNIR band index to use for brightness detection.
    threshold : float, optional
        Brightness threshold fraction (default 0.9).

    Returns
    -------
    bbox : tuple[int, int, int, int]
        Bounding box coordinates (x1, y1, x2, y2)
    panel_height : int
        Height/width of detected panel region (square).
    """

    Y, X, B = cube.shape
    sensor_type = "VNIR" if vnird_bbox is None else "SWIR"
    print(f"[detect_panel_region] ({sensor_type}) cube shape (Y,X,B)={cube.shape}")

    # =====================================================
    # VNIR PANEL DETECTION
    # =====================================================
    if sensor_type == "VNIR":
        if band_index is None:
            band_index = B // 2

        img = cube[:, :, band_index].astype(np.float32)
        norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        thr = threshold or 0.9
        mask = norm >= thr

        coords = np.argwhere(mask)
        if coords.size == 0:
            cy, cx = Y // 2, X // 2
            bbox = (cx - 5, cy - 5, cx + 5, cy + 5)
            print("[VNIR] fallback bbox:", bbox)
            return bbox, 10

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        panel_height = int(max(1, y_max - y_min))
        bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
        print(f"[VNIR] detected bbox: {bbox}, height={panel_height}")
        return bbox, panel_height

    # =====================================================
    # SWIR PANEL DETECTION (auto VNIR→SWIR scaling)
    # =====================================================
    print("[SWIR] Starting geometric VNIR→SWIR mapping...")

    if vnird_bbox is None or vnir_file is None or swir_file is None:
        raise ValueError("[SWIR] Missing required arguments: vnird_bbox, vnir_file, or swir_file")

    # --- Parse shapes from filenames ---
    vnir_shape = parse_shape_from_filename(vnir_file)
    swir_shape = parse_shape_from_filename(swir_file)
    h_vnir, w_vnir, _ = vnir_shape
    h_swir, w_swir, _ = swir_shape

    # --- Compute scaling ratios ---
    scale_x = w_swir / w_vnir
    scale_y = h_swir / h_vnir
    print(f"[SWIR] scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")

    # --- Scale VNIR bbox into SWIR coordinates ---
    x1v, y1v, x2v, y2v = vnird_bbox
    x1s = int(x1v * scale_x)
    x2s = int(x2v * scale_x)
    y1s = int(y1v * scale_y)
    y2s = int(y2v * scale_y)
    panel_height_s = y2s - y1s

    print(f"[SWIR] initial scaled bbox=({x1s},{y1s},{x2s},{y2s}), height={panel_height_s}")

    # --- Expand ±15% search window ---
    pad_x = int(0.15 * (x2s - x1s))
    pad_y = int(0.15 * (y2s - y1s))
    x_min = max(0, x1s - pad_x)
    x_max = min(w_swir, x2s + pad_x)
    y_min = max(0, y1s - pad_y)
    y_max = min(h_swir, y2s + pad_y)
    print(f"[SWIR] expanded search: x[{x_min}:{x_max}], y[{y_min}:{y_max}]")

    # --- Mean brightness map ---
    region = cube[y_min:y_max, x_min:x_max, :]
    brightness = region.mean(axis=2)
    norm_brightness = (brightness - brightness.min()) / (brightness.ptp() + 1e-6)

    bright_thresh = np.percentile(norm_brightness, 90)
    bright_mask = norm_brightness >= bright_thresh

    if not np.any(bright_mask):
        print("[SWIR] No bright region found → fallback to scaled VNIR bbox")
        return (x1s, y1s, x2s, y2s), panel_height_s

    # --- Centroid of bright region ---
    y_idx, x_idx = np.where(bright_mask)
    cx = int(np.mean(x_idx)) + x_min
    cy = int(np.mean(y_idx)) + y_min

    # --- Square box around centroid ---
    half_side = panel_height_s // 2
    x_min_box = max(0, cx - half_side)
    x_max_box = min(w_swir, cx + half_side)
    y_min_box = max(0, cy - half_side)
    y_max_box = min(h_swir, cy + half_side)

    # --- Enforce perfect square ---
    side = min(x_max_box - x_min_box, y_max_box - y_min_box)
    x_max_box = x_min_box + side
    y_max_box = y_min_box + side

    print(f"[SWIR] final bbox=({x_min_box},{y_min_box},{x_max_box},{y_max_box}), side={side}")
    return (x_min_box, y_min_box, x_max_box, y_max_box), side

# ============================================================
# 3️⃣ Visualize Scaled VNIR Panel on SWIR (manual offset & height reduction)
# ============================================================
def visualize_panel_detection(cube, wavelengths, panel_bbox, save_path=None, sensor_type="VNIR"):
    """
    Visualize the panel region overlay on RGB (VNIR) or grayscale (SWIR) image.
    """
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
    from pathlib import Path

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


def overlay_scaled_vnir_on_swir(cube_swir, panel_bbox_vnir, vnir_file, swir_file,
                                wl_target=1550, x_offset=0, y_offset=0,
                                height_reduction_factor=0.65, save_path=None):
    """
    Scale a VNIR panel bounding box to SWIR coordinates, apply manual offsets and
    reduce height by a given factor. Plot overlay on SWIR grayscale.

    Parameters
    ----------
    cube_swir : np.ndarray
        SWIR datacube (Y, X, Bands)
    panel_bbox_vnir : tuple[int]
        VNIR bounding box (x1, y1, x2, y2)
    vnir_file, swir_file : Path
        Filenames for auto shape detection
    wl_target : int
        SWIR wavelength for visualization
    x_offset, y_offset : float
        Manual offsets in pixels
    height_reduction_factor : float
        Fraction of height to retain
    save_path : Path, optional
        Path to save the overlay image

    Returns
    -------
    scaled_bbox : tuple[int]
        Final SWIR panel bounding box
    """
    h_vnir, w_vnir, _ = parse_shape_from_filename(vnir_file)
    h_swir, w_swir, _ = parse_shape_from_filename(swir_file)

    # Scale VNIR bbox to SWIR
    x1v, y1v, x2v, y2v = panel_bbox_vnir
    x1s = int(x1v * w_swir / w_vnir + x_offset)
    x2s = int(x2v * w_swir / w_vnir + x_offset)
    y1s = int(y1v * h_swir / h_vnir + y_offset)
    y2s = int(y2v * h_swir / h_vnir + y_offset)

    # Reduce height
    y_center = (y1s + y2s) / 2
    half_height = (y2s - y1s) * height_reduction_factor / 2
    y1s = int(y_center - half_height)
    y2s = int(y_center + half_height)

    scaled_bbox = (x1s, y1s, x2s, y2s)
    print(f"[overlay_scaled_vnir_on_swir] VNIR bbox {panel_bbox_vnir} → SWIR scaled bbox {scaled_bbox} "
          f"(offset X={x_offset}, Y={y_offset}, height_factor={height_reduction_factor})")

    # Grab grayscale at target wavelength
    wavelengths = np.linspace(1000, 2500, cube_swir.shape[2])
    idx = np.argmin(np.abs(wavelengths - wl_target))
    gray = cube_swir[:, :, idx]

    # Plot overlay
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(gray, cmap="gray")
    rect = Rectangle((x1s, y1s), x2s - x1s, y2s - y1s, linewidth=0.4, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(f"SWIR {wl_target}nm with panel overlay")
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved overlay → {Path(save_path).name}")
    plt.show()

    return scaled_bbox

##################################
## Additional - SWIR panel refinement function
##################################

import json
import numpy as np
from matplotlib.patches import Rectangle
from pathlib import Path

def refine_swir_panel_bbox(cube_swir, prev_bbox, vnir_file=None, swir_file=None,
                           wl_pct=(0.30, 0.55, 0.75), safe_shift_nm=-50,
                           search_expand_factor=10, box_side_factor=1.0,
                           save_path=None, debug=False):
    """
    Refine a SWIR panel bbox by building a 3-wavelength aggregate, searching
    inside a 10x region around prev_bbox centroid for the brightest pixel,
    and returning a square refined bbox centered on that brightest location.

    Args:
        cube_swir : np.ndarray (Y, X, B)
            SWIR datacube already in display orientation (i.e. same as used for plotting).
        prev_bbox : tuple (x1, y1, x2, y2)
            Prior SWIR bbox (global coordinates in cube_swir space).
        vnir_file, swir_file : Path or str (optional)
            Used only if you want to derive wavelengths from filenames (not required).
        wl_pct : tuple of floats
            Fractions across 1000-2500 nm for the three test wavelengths.
        safe_shift_nm : int
            If a chosen wavelength falls on a strong absorption, shift it (nm).
        search_expand_factor : int
            Factor to expand prev_bbox dimensions (default 10).
        box_side_factor : float
            Final box side = panel_height_s * box_side_factor
        save_path : Path or str
            Where to save diagnostic PNG (optional).
        debug : bool
            If True, prints extra diagnostic info.

    Returns:
        refined_bbox (x1, y1, x2, y2), sub_cube (the 3-band subcube used for detection)
    """
    Y, X, B = cube_swir.shape
    # --- build wavelengths (cube bands mapped to 1000-2500 nm) ---
    wavelengths = np.linspace(1000, 2500, B)

    # choose three wavelengths as per wl_pct
    chosen_wls = []
    for p in wl_pct:
        w = 1000 + p * (2500 - 1000)
        # avoid harsh absorption windows around ~1400 nm and ~1900-2000/2400 nm
        # simple safety: if w within 1390-1420 shift by safe_shift_nm
        if 1390 <= w <= 1420 or 1980 <= w <= 2020 or 2380 <= w <= 2420:
            w = w + safe_shift_nm
        chosen_wls.append(w)
    if debug:
        print("[refine] chosen wavelengths:", chosen_wls)

    idxs = [int(np.argmin(np.abs(wavelengths - w))) for w in chosen_wls]
    if debug:
        print("[refine] band indices used:", idxs)

    # Subcube composite (Y, X, 3)
    sub_rgb = np.stack([cube_swir[:, :, i] for i in idxs], axis=-1).astype(np.float32)

    # Build the big search window around prev_bbox centroid (in global coords)
    x1_p, y1_p, x2_p, y2_p = prev_bbox
    cx = (x1_p + x2_p) / 2.0
    cy = (y1_p + y2_p) / 2.0
    w0 = (x2_p - x1_p)
    h0 = (y2_p - y1_p)

    half_w = int((w0 * search_expand_factor) / 2)
    half_h = int((h0 * search_expand_factor) / 2)
    x_min = max(0, int(cx - half_w))
    x_max = min(X, int(cx + half_w))
    y_min = max(0, int(cy - half_h))
    y_max = min(Y, int(cy + half_h))

    if debug:
        print(f"[refine] search box global coords x[{x_min}:{x_max}] y[{y_min}:{y_max}]")

    # Crop the sub-area for detection
    crop = sub_rgb[y_min:y_max, x_min:x_max, :]  # local coords
    if crop.size == 0:
        raise ValueError("Empty crop for refinement — check prev_bbox and cube sizes.")

    # Create a brightness map by summing (or mean) across the 3 bands
    brightness = crop.mean(axis=2)  # (y_local, x_local)

    # Find the brightest pixel via argmax — stable and unambiguous
    flat_idx = np.nanargmax(brightness)
    local_y, local_x = np.unravel_index(flat_idx, brightness.shape)

    # Convert local -> global coordinates (important: no y-flip here)
    global_x = x_min + int(local_x)
    global_y = y_min + int(local_y)
    if debug:
        print(f"[refine] brightest local (x,y) = ({local_x},{local_y}) -> global (x,y)=({global_x},{global_y})")

    # Build refined square bbox centered on (global_x, global_y)
    # Use prev panel height (or fallback to median of w0/h0)
    nominal_side = max(1, int(((y2_p - y1_p) + (x2_p - x1_p)) / 2.0 * box_side_factor))
    half_side = nominal_side // 2
    x1_r = max(0, global_x - half_side)
    x2_r = min(X, global_x + half_side)
    y1_r = max(0, global_y - half_side)
    y2_r = min(Y, global_y + half_side)

    refined_bbox = (int(x1_r), int(y1_r), int(x2_r), int(y2_r))

    # Diagnostic plotting and saving
    if save_path is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        # build a normalized RGB for display from crop (stretch to 0-1)
        crop_disp = crop.copy()
        mn = np.nanmin(crop_disp)
        mx = np.nanmax(crop_disp)
        crop_disp = (crop_disp - mn) / (mx - mn + 1e-9)
        # But we want to display the full image context (not just crop): create full rgb
        full_rgb = np.zeros((Y, X, 3), dtype=np.float32)
        for k in range(3):
            band = cube_swir[:, :, idxs[k]].astype(np.float32)
            mn_b, mx_b = np.nanmin(band), np.nanmax(band)
            full_rgb[:, :, k] = (band - mn_b) / (mx_b - mn_b + 1e-9)
        ax.imshow(np.clip(full_rgb, 0, 1))

        # draw 10x search region (yellow)
        ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               linewidth=1.0, edgecolor='yellow', facecolor='none', zorder=2))
        # draw previous bbox (green)
        ax.add_patch(Rectangle((x1_p, y1_p), x2_p - x1_p, y2_p - y1_p,
                               linewidth=1.2, edgecolor='lime', facecolor='none', zorder=3))
        # draw refined bbox (red)
        ax.add_patch(Rectangle((refined_bbox[0], refined_bbox[1]),
                               refined_bbox[2] - refined_bbox[0],
                               refined_bbox[3] - refined_bbox[1],
                               linewidth=2.0, edgecolor='red', facecolor='none', zorder=4))
        # mark bright pixel
        ax.scatter([global_x], [global_y], marker='s', s=40, c='green', zorder=5)

        ax.set_title(f"SWIR panel refinement ({int(chosen_wls[0])}/{int(chosen_wls[1])}/{int(chosen_wls[2])} nm composite)")
        ax.axis('off')
        plt.tight_layout()

        out_png = Path(save_path)
        fig.savefig(out_png, dpi=200)
        if debug:
            print(f"[refine] saved overlay image {out_png}")
        plt.close(fig)

        # also save refined bbox as JSON alongside the overlay
        json_path = out_png.with_suffix(".json")
        with open(json_path, "w") as fh:
            json.dump({"refined_bbox": refined_bbox, "prev_bbox": prev_bbox,
                       "wavelengths": chosen_wls}, fh, indent=2)
        if debug:
            print(f"[refine] saved bbox json {json_path}")

    return refined_bbox, crop


# ============================================================
# 4️⃣ Compute Panel Reflectance
# ============================================================

def compute_panel_reflectance(cube, panel_bbox, panel_cal_file, wavelengths=None):
    x1, y1, x2, y2 = panel_bbox
    panel_region = cube[y1:y2, x1:x2, :]
    mean_panel_signal = panel_region.mean(axis=(0, 1))
    cal_data = np.loadtxt(panel_cal_file, comments="#", usecols=(0, 1))
    cal_wvl, cal_refl = cal_data[:, 0], cal_data[:, 1]
    if wavelengths is None:
        wavelengths = np.linspace(cal_wvl.min(), cal_wvl.max(), cube.shape[-1])
    interp_reflectance = np.interp(wavelengths, cal_wvl, cal_refl)
    return mean_panel_signal, interp_reflectance


# ============================================================
# 5️⃣ Apply Reflectance Correction
# ============================================================

def apply_reflectance_correction(cube, mean_panel_signal, interp_panel_reflectance, panel_bbox):
    safe_panel = np.where(mean_panel_signal == 0, np.nan, mean_panel_signal)
    correction_factor = interp_panel_reflectance / safe_panel
    corrected_cube = cube * correction_factor[np.newaxis, np.newaxis, :]
    x1, y1, x2, y2 = panel_bbox
    corrected_cube[y1:y2, x1:x2, :] = np.nan
    return corrected_cube


# ============================================================
# 6️⃣ Export
# ============================================================

def export_reflectance_data(cube, output_path):
    output_path = Path(output_path)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".npy")
    np.save(output_path, cube)
    print(f"Exported reflectance cube → {output_path.name}")

# ============================================================
# 7️⃣ NetCDF export (improved) + panel extraction (fixed)
# ============================================================
import xarray as xr

def export_reflectance_to_netcdf(cube, wavelengths, output_path, var_name="reflectance"):
    """
    Export a 3D cube (Y,X,Bands) to NetCDF with wavelength coordinate.
    cube shape must be (y, x, bands).
    """
    output_path = Path(output_path)
    if output_path.suffix != ".nc":
        output_path = output_path.with_suffix(".nc")

    # Build xarray dataset with dims (y, x, wavelength)
    ds = xr.Dataset(
        {var_name: (("y", "x", "wavelength"), cube.astype(np.float32))},
        coords={
            "y": np.arange(cube.shape[0]),
            "x": np.arange(cube.shape[1]),
            "wavelength": wavelengths
        }
    )
    ds.to_netcdf(output_path)
    print(f"Saved NetCDF reflectance → {output_path.name}")


def compute_panel_reflectance(cube, panel_bbox, panel_cal_file, wavelengths=None):
    """
    Return mean_panel_signal (raw instrument units) and interp_panel_reflectance (known panel reflectance).
    This function does NOT apply the correction to the whole cube.
    """
    x1, y1, x2, y2 = panel_bbox
    panel_region = cube[y1:y2, x1:x2, :]  # shape = (yp, xp, bands)
    mean_panel_signal = np.nanmean(panel_region, axis=(0, 1))  # (bands,)

    # load calibration file (wavelength, reflectance)
    cal_data = np.loadtxt(panel_cal_file, comments="#", usecols=(0, 1))
    cal_wvl, cal_refl = cal_data[:, 0], cal_data[:, 1]

    # if wavelengths provided, interpolate calibration to cube bands
    if wavelengths is None:
        wavelengths = np.linspace(cal_wvl.min(), cal_wvl.max(), cube.shape[-1])

    interp_reflectance = np.interp(wavelengths, cal_wvl, cal_refl)
    return mean_panel_signal, interp_reflectance


def extract_panel_reflectance_cube(cube, panel_bbox, panel_cal_file, wavelengths=None):
    """
    Extract the panel region from cube and convert it to reflectance using
    the panel calibration file.

    Returns:
        panel_reflectance_cube: shape (yp, xp, bands) float32
        mean_panel_signal: (bands,) average raw signal from panel pixels
        interp_reflectance: (bands,) known panel reflectance (interpolated)
    """
    x1, y1, x2, y2 = panel_bbox
    panel_region = cube[y1:y2, x1:x2, :].astype(np.float32)  # (yp, xp, bands)

    if panel_region.size == 0:
        raise ValueError("Panel region is empty. Check panel_bbox coordinates.")

    # compute mean raw signal across panel pixels per band
    mean_panel_signal = np.nanmean(panel_region, axis=(0, 1))  # (bands,)

    # interpolate calibration reflectance
    cal_data = np.loadtxt(panel_cal_file, comments="#", usecols=(0, 1))
    cal_wvl, cal_refl = cal_data[:, 0], cal_data[:, 1]
    if wavelengths is None:
        wavelengths = np.linspace(cal_wvl.min(), cal_wvl.max(), cube.shape[-1])
    interp_reflectance = np.interp(wavelengths, cal_wvl, cal_refl)  # (bands,)

    # protect against zeros
    safe_mean = np.where(mean_panel_signal == 0, np.nan, mean_panel_signal)

    # correction factor: multiply raw -> reflectance
    correction_factor = interp_reflectance / safe_mean  # shape (bands,)

    # Apply per-band correction to the panel pixels (broadcast across spatial dims)
    panel_reflectance = panel_region * correction_factor[np.newaxis, np.newaxis, :]

    # diagnostics: mean of converted panel pixels should ~ interp_reflectance
    panel_mean_after = np.nanmean(panel_reflectance, axis=(0, 1))
    # Compute simple difference stats
    diff = panel_mean_after - interp_reflectance
    max_abs_diff = np.nanmax(np.abs(diff))
    mean_rel_error = np.nanmean(np.abs(diff) / (np.where(interp_reflectance == 0, 1e-12, interp_reflectance)))

    print("=== Panel conversion diagnostics ===")
    print(f"Panel pixel block shape: {panel_reflectance.shape}")
    print(f"Mean panel reflectance (post-conversion) -> first 5 bands: {panel_mean_after[:5]}")
    print(f"Expected interp reflectance -> first 5 bands: {interp_reflectance[:5]}")
    print(f"Max abs difference: {max_abs_diff:.6f}")
    print(f"Mean relative error: {mean_rel_error:.6f}")
    if max_abs_diff > 1e-2 or mean_rel_error > 0.01:
        print("⚠️  Warning: panel mean after conversion differs from calibration by more than tolerance.")
        print("  - This may indicate an incorrect panel bbox, wrong calibration file, or additional instrument scaling required.")

    return panel_reflectance.astype(np.float32), mean_panel_signal.astype(np.float32), interp_reflectance.astype(np.float32)


def extract_region_around_panel(cube, panel_bbox, interp_panel_reflectance=None, panel_mean_signal=None, x_fraction=0.1):
    """
    Extract sub-regions to the left and right of the panel (+/- x_fraction*panel_width)
    and convert them to reflectance using either:
      - interp_panel_reflectance (preferred), OR
      - panel_mean_signal (raw) together with interpolated panel reflectance.

    Returns:
        combined_reflectance: concatenated reflectance regions (y, x_total, bands)
    """
    Y, X, B = cube.shape
    x1, y1, x2, y2 = panel_bbox
    panel_width = x2 - x1
    x_offset = int(panel_width * x_fraction)

    # left region
    x_start_left = max(0, x1 - x_offset)
    x_end_left = x1
    # right region
    x_start_right = x2
    x_end_right = min(X, x2 + x_offset)

    regions = []
    for xs, xe in [(x_start_left, x_end_left), (x_start_right, x_end_right)]:
        if xe <= xs:
            # empty region; create empty array
            regions.append(np.zeros((Y, 0, B), dtype=np.float32))
            continue
        region_cube = cube[:, xs:xe, :].astype(np.float32)

        # determine correction factor per band
        if interp_panel_reflectance is not None and panel_mean_signal is not None:
            safe_mean = np.where(panel_mean_signal == 0, np.nan, panel_mean_signal)
            corr = interp_panel_reflectance / safe_mean
        elif interp_panel_reflectance is not None:
            # we need mean raw signal estimate — approximate with region mean? less ideal.
            mean_est = np.nanmean(region_cube, axis=(0, 1))
            safe_mean = np.where(mean_est == 0, np.nan, mean_est)
            corr = interp_panel_reflectance / safe_mean
            print("⚠️  Using region mean as approximation for correction factor (not recommended).")
        else:
            raise ValueError("Provide interp_panel_reflectance (preferred) and panel_mean_signal.")

        region_reflectance = region_cube * corr[np.newaxis, np.newaxis, :]
        regions.append(region_reflectance)

    # concatenate along x axis (axis=1)
    combined_reflectance = np.concatenate(regions, axis=1)
    return combined_reflectance
