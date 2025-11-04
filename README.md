# Hyper_spectra_automation

Code to process ANU Forest Spectrometer outputs. This project is primarily written in Python and consists of a processing pipeline (`panel_reflectance_pipeline.py`) and a Jupyter Notebook that orchestrates the workflow and visualizes the data. The code processes multiple hyperspectral data cubes, performs automatic calibration panel detection, and converts raw spectral data to reflectance, integrating scan metadata along the way.  

## Features
- Recursive processing of hyperspectral `.bin` files across date/instrument subfolders.  
- Automatic calibration panel detection (VNIR & SWIR).  
- Reflectance calculation using calibration panels.  
- Export of results as NetCDF (`.nc`) files and CSV summaries.  
- Interactive panel region adjustment for verification and correction.  
- Visualization and verification plots of processed data.

## Workflow Overview

The workflow is implemented in a Jupyter Notebook and follows these steps:

1. **Imports & Configuration Paths**  
   - Load required Python libraries and define paths to raw data, calibration files, and output directories.

2. **Pipeline Module Import**  
   - Import functions from `panel_reflectance_pipeline.py` for data processing and analysis.

3. **Test Run: Panel Detection**  
   - Run a first test of panel selection to display detected panel regions.

4. **Calibration File Check**  
   - Verify reflectance calibration panel data and parameters.

5. **Recursive Multi-Folder Processing**  
   - Walk through all date/instrument folders and process each SWIR/VNIR `.bin` file automatically.

6. **Interactive Panel Polygon Editor (Optional)**  
   - Adjust or redraw detected panel regions visually and recompute averages.

7. **SWIR Panel Detection Refinement**  
   - Revise SWIR panel detection to ensure robust pixel identification.

8. **Panel Reflectance Extraction**  
   - Compute and save reflectance values as NetCDF files.

9. **Vegetation Reflectance Extraction**  
   - Extract regions around the panel and save averaged vegetation reflectance as NetCDF files.

10. **Output Review & Export**  
    - Inspect `.nc` outputs, export CSVs, and generate verification plots.

## Usage

1. Clone this repository:

```bash
git clone https://github.com/Eric-git-999/Hyper_spectra_automation.git
cd Hyper_spectra_automation
```

2. Open the main Jupyter Notebook:
```bash
jupyter notebook ANU_Forest_Spectrometer_Workflow.ipynb
```

3. Configure paths and run each cell sequentially.

4. Optional: Adjust panel regions interactively as needed.

## Dependencies

This workflow requires the following Python libraries:

- Python 3.8 or higher
- NumPy
- Pandas
- Matplotlib
- xarray
- netCDF4
- Any other libraries listed in the notebook or in `requirements.txt`

## Output

After running the workflow, the following outputs will be generated:

- Processed hyperspectral data saved as NetCDF (`.nc`) files
- Averaged vegetation reflectance outputs
- CSV summaries and verification plots for visual inspection
