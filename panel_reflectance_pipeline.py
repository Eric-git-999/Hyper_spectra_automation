## This code was developed by Eric Hay (2025) as part of workflow and automated pipeline
## work for the ANU Fenner School BRCoE

## The below pipeline works to operate on the ANU Forest Spectroscope outputs.
## These are from the Headwall Nano and Headwall SWIR hyperspectral / pushbroom sensors.

## The code works assuming the scan captures a calibrated reflectance panel, and that calibration data are provided.

## Format of panel calibration file: Wavelength and signal columns, numbers only.
## An example panel file is provided (LARGE_PANEL.txt) that was used for ANU processing.

"""
panel_reflectance_pipeline.py
-----------------------------
Enhanced module for VNIR/SWIR spectral datacube reflectance processing.

Updates (4.11.25):
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
# 2️⃣ Detect Panel Region - VNIR 
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


# ============================================================
# 3️⃣ Visualize Scaled panel detection
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


# ============================================================
# 4️⃣ SWIR single function panel detection
# ============================================================

def detect_swir_panel_single_pass(cube_swir, wl_pct=(0.30, 0.55, 0.75),
                                  safe_shift_nm=-50, box_side_factor=1.0,
                                  save_path=None, debug=False):
    """
    Single-pass SWIR panel detection using 3-band subcube across the whole image.
    
    Args:
        cube_swir : np.ndarray (Y, X, B)
            SWIR datacube in display orientation
        wl_pct : tuple of floats
            Fraction across 1000-2500 nm to choose 3 wavelengths
        safe_shift_nm : int
            Shift if chosen wavelength falls in strong absorption
        box_side_factor : float
            Final box side = panel_height_s * box_side_factor
        save_path : Path or str, optional
            Path to save diagnostic PNG
        debug : bool
            Print debug information
            
    Returns:
        refined_bbox (x1, y1, x2, y2)
    """
    Y, X, B = cube_swir.shape
    wavelengths = np.linspace(1000, 2500, B)
    
    # Select 3 safe wavelengths
    chosen_wls = []
    for p in wl_pct:
        w = 1000 + p*(2500-1000)
        if 1390 <= w <= 1420 or 1980 <= w <= 2020 or 2380 <= w <= 2420:
            w += safe_shift_nm
        chosen_wls.append(w)
    idxs = [int(np.argmin(np.abs(wavelengths - w))) for w in chosen_wls]
    if debug: print("[SWIR single-pass] chosen wavelengths:", chosen_wls, "band indices:", idxs)
    
    # Build subcube (Y, X, 3)
    subcube = np.stack([cube_swir[:, :, i] for i in idxs], axis=-1).astype(np.float32)
    
    # Brightness map
    brightness = subcube.mean(axis=2)
    
    # Detect brightest pixel
    flat_idx = np.nanargmax(brightness)
    cy, cx = np.unravel_index(flat_idx, brightness.shape)
    if debug: print(f"[SWIR single-pass] brightest pixel at (x,y)=({cx},{cy})")
    
    # Determine nominal panel size (use median of width/height or arbitrary fraction)
    # Here, use 5% of image width as rough panel size
    nominal_side = int(min(X, Y) * 0.05 * box_side_factor)
    half_side = nominal_side // 2
    
    # Build square bbox
    x1 = max(0, cx - half_side)
    x2 = min(X, cx + half_side)
    y1 = max(0, cy - half_side)
    y2 = min(Y, cy + half_side)
    refined_bbox = (x1, y1, x2, y2)
    
    # Optional diagnostic plot
    if save_path is not None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        fig, ax = plt.subplots(figsize=(10, 5))
        # normalized 3-band RGB
        rgb_disp = subcube.copy()
        for k in range(3):
            mn, mx = np.nanmin(rgb_disp[:,:,k]), np.nanmax(rgb_disp[:,:,k])
            rgb_disp[:,:,k] = (rgb_disp[:,:,k]-mn)/(mx-mn+1e-9)
        ax.imshow(np.clip(rgb_disp,0,1))
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor='red', facecolor='none'))
        ax.scatter([cx],[cy], marker='x', c='lime')
        ax.set_title("SWIR panel single-pass panel detection")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        if debug: print(f"[SWIR single-pass] saved diagnostic overlay: {save_path}")
    
    return refined_bbox

def swir_composite(cube, wl_pct=(0.30, 0.55, 0.75)):
    h, w, bands = cube.shape
    b1 = int(bands * wl_pct[0])
    b2 = int(bands * wl_pct[1])
    b3 = int(bands * wl_pct[2])
    comp = np.stack([
        cube[:, :, b1],
        cube[:, :, b2],
        cube[:, :, b3]
    ], axis=-1)
    
    # normalize to 0-255 for display
    comp = comp.astype(float)
    comp -= comp.min()
    comp /= comp.max()
    comp = (comp * 255).astype(np.uint8)
    return comp


# ============================================================
# 5️⃣ Compute Panel Reflectance
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
# 6️⃣ Apply Reflectance Correction
# ============================================================

def apply_reflectance_correction(cube, mean_panel_signal, interp_panel_reflectance, panel_bbox):
    safe_panel = np.where(mean_panel_signal == 0, np.nan, mean_panel_signal)
    correction_factor = interp_panel_reflectance / safe_panel
    corrected_cube = cube * correction_factor[np.newaxis, np.newaxis, :]
    x1, y1, x2, y2 = panel_bbox
    corrected_cube[y1:y2, x1:x2, :] = np.nan
    return corrected_cube


# ============================================================
# 7️⃣ Export
# ============================================================

def export_reflectance_data(cube, output_path):
    output_path = Path(output_path)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".npy")
    np.save(output_path, cube)
    print(f"Exported reflectance cube → {output_path.name}")

# ============================================================
# 8️⃣ NetCDF export (improved) + panel extraction (fixed)
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
