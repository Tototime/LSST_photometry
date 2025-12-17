# LSST Photometry

This repository provides tools and studies for analyzing LSST photometric data, focusing on star and galaxy photometry. It includes notebooks for data visualization, photometric error analysis, and catalog exploration, along with Python modules for reusable methods. The primary use case revolves around the LSST science pipeline products of DP1 and DC2.

## Repository Structure

### `notebooks/`
Contains notebooks for specific LSST photometric topics:
- **`dc2_star_photometry.ipynb`**: Analysis of DC2 photometry catalogs.
- **`dp1_star_photometry.ipynb`**: Analysis of DP1 photometry catalogs.
- **`synthetic_error_from_catalog.ipynb`**: Demonstrates random photometric error generation from empirical distributions.

### `notebooks_for_LSST_cloud/`
Notebooks tailored for the LSST Cloud services:
- **`butler_demo.ipynb`**: Example access to LSST Butler data using Python APIs.
- **`Custom1_Lephare_custom_tables.ipynb`**: Builds tables compatible with LePHARE for galaxies and stars.
- **`Custom2_dowload_and_derive_stellar_photometry.ipynb`**: Downloads and filters stellar photometry data.
- **`Custom3_retrieve_maglim_maps.ipynb`**: Retrieves magnitude limit maps from Butler and converts to usable formats.

### `src/`
Python modules providing reusable components:
- **`photometry_plots.py`**: Helper functions for generating photometric plots.
- **`photometry_errors.py`**: Code for error modeling and sampling.