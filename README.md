# Hyper_spectra_automation

Code to process ANU Forest Spectrometer outputs. This project is primarily written in Python and consists of Python helper functions (`panel_reflectance_pipeline.py`) and a Jupyter Notebook (`auto_spectra.ipynb`) that orchestrates the workflow and visualizes the data. The code processes multiple hyperspectral data cubes, performs automatic calibration panel detection, and converts raw spectral data to reflectance, integrating scan metadata along the way.  

## Features
- Recursive processing of hyperspectral `.bin` files across date/instrument subfolders.  
- Automatic calibration panel detection (VNIR & SWIR).  
- Reflectance calculation using calibration panels.  
- Export of results as NetCDF (`.nc`) files and CSV summaries.  
- Interactive selection region adjustment for verification and correction.  
- Visualization and verification plots of processed data.

## Workflow Overview

The workflow is implemented in a Jupyter Notebook and follows these steps:

1. **Imports & Configuration Paths**  
   - Load required Python libraries and define paths to raw data, calibration files, and output directories.

2. **Automated panel detection workflow**  
   - Import functions from `panel_reflectance_pipeline.py` for data processing and analysis.
   - Iterates through scan subfolders, working on the instrument output (reordered .bin datacube files). The detection is based on brightness, pixel thresholding and for SWIR, a subset region based on the VNIR detection which is more robust.

3. **Panel-based Reflectance Conversion Pipeline - VNIR & SWIR**  
   - Uses the detected panel regions and for both VNIR and SWIR, using the panel calibration file "LARGE_PANEL.txt".
   - Computes the median reflectance value for the panel region, saves this as a .csv, and processes the original datacube to a reflectance datacube.

4. **Panel Detection Recovery Workflow (Manual Reflectance Assignment)**  
   - exists as a catch for failed panel detection files.
   - It reads the error report (.txt, if exists) to show which files need to have a panel region applied and allows the use of an adjacent scan (that had successful panel detection) to use that reflectance median for either VNIR / SWIR and apply datacube reflectance conversion based on the supplied reflectance panel data.

5. **Interactive NetCDF Reflectance Cube Viewer**  
   - Walk through all date/instrument folders and process each SWIR/VNIR `.bin` file automatically.

6. **Interactive Panel Polygon Editor (Optional)**  
   - Allows you to specify a netCDF reflectance cube to examine how it looks.

## Usage

1. Clone this repository:

```bash
git clone https://github.com/Eric-git-999/Hyper_spectra_automation.git
cd Hyper_spectra_automation
```

2. Open the main Jupyter Notebook:
```bash
jupyter notebook Auto_spectra.ipynb.ipynb
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
- os
- collections
- skimage
- scipy.ndimage
- IPython.display
- importlib
- gc
- json
- Any other libraries listed in the notebook or in `requirements.txt`

## Output

After running the workflow, the following outputs will be generated:

- Processed hyperspectral reflectance datacubes saved as NetCDF (`.nc`) files, for both VNIR and SWIR cubes
- Median panel reflectance values, saved as a .csv for each scan
- Verification plots for visual inspection, showing panel detection regions and representations of the imported datacubes with panel detection overlay

## Example Notebook Outputs

![VNIR Panel detection 1](images/VNIR1.png)
![SWIR Panel detection 1](images/SWIR1.png)
![VNIR Panel detection 2](images/VNIR2.png)
![SWIR Panel detection 2](images/SWIR2.png)
![VNIR Panel detection 3](images/VNIR3.png)
![SWIR Panel detection 3](images/SWIR3.png)
![VNIR Panel reflectance 1](images/VNIR_p1.png)
![SWIR Panel reflectance 1](images/SWIR_p1.png)
![SWIR Panel reflectance cube display](images/SWIR_p2.png)

