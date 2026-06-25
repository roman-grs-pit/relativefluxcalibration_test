# Roman WFI Relative Flux Calibration

Simulation and chi-squared minimization framework for testing wavelength-dependent relative flux calibration of the Nancy Grace Roman Space Telescope Wide Field Imager (WFI) grism. Implements the ubercal scheme from Padmanabhan et al. 2008 and Markovic et al. 2017.

## Overview

Simulates a multi-pointing Roman survey, places stars from a mock catalog onto detector footprints, computes grism dispersion traces per star, and recovers per-exposure flux offsets (`k_i`) and a wavelength-dependent flat-field correction (`A_λ`) via L-BFGS-B minimization. Residuals are saved as calibration precision metrics.

**Main script:** `FinalRomanCalibration_v5.py`

## Pipeline Steps

1. Load star catalog (`~/project/sim_star_cat_galacticus.ecsv`) and apply magnitude cut
2. Build survey pointing grid with 4 position angles × 2 dithers per tile
3. Compute WFI detector footprints using `pysiaf`
4. Map each star's grism trace onto the detector using the Roman optical model
5. Simulate observations with injected per-exposure offsets (`k_i`, σ=0.5 mag), vignetting (`A_λ`), grism throughput, and photon noise from Pandeia
6. Recover `k_i` and `A_λ` jointly via chi2 minimization (`chi2J_vec_onestep_kA`)
7. Recover wavelength-dependent residual `r_λ` in a second pass (`chi2J_vec_onestep_r`)
8. Save residual RMS to `wficalibtest_sigmas/`

## Dependencies

Python packages: numpy, scipy, pandas, matplotlib, astropy, pysiaf, tqdm, yaml

## Required Files

**Python modules**
```
FinalRomanCalibration_v5.py
observing-program/py/chi2_func.py
observing-program/py/overlap_cal_func.py
observing-program/py/footprintutils.py
observing-program/py/roman_coords_transform.py
observing-program/py/grism_dispersion.py
grism_sim/py/optical_model.py
```

**Data**
```
observing-program/data/roman_wfi_detector_coords.json
grism_sim/data/Roman_grism_OpticalModel_v0.6.yaml
grism_sim/data/Roman.det*.conf          # 18 per-detector dispersion configs
grism_sim/data/SCA*_sens_*.fits         # 18 per-SCA sensitivity curves (consider Git LFS)
Grism total throughput - output data.csv
mag_data/flat_spec_<mag>_195.txt        # one file per 0.5-mag step in your magnitude range
```

**External (not in repo)**
```
~/project/sim_star_cat_galacticus.ecsv  # star catalog; path is hardcoded in the script
```

**Output directory (must exist before running)**
```
wficalibtest_sigmas/
```

**Cluster scripts (optional)**
```
run_wficalib_v5.slurm
run_wficalib_v5_array.slurm
```

## Running

**Arguments:** `minmag maxmag dithersize`

```bash
# Single run (magnitudes 14–17, dither scale 1.0)
python FinalRomanCalibration_v5.py 14 17 1.0

# SLURM single job
sbatch run_wficalib_v5.slurm 14 17 1.0

# SLURM array sweep over dithersize 0.00–4.00 in steps of 0.01
sbatch run_wficalib_v5_array.slurm
```

Output: `wficalibtest_sigmas/wficalibtest_v5_<minmag>_<maxmag>_<dithersize>_195.txt`
containing four values: `[σ_f,  σ_f(5σ-clipped),  σ_fE,  σ_fE(5σ-clipped)]` in magnitudes.

## Key Parameters

| Parameter | Value | Description |
|---|---|---|
| `survey` | 3 | 4 PA × 2 dither pattern |
| `sigk` | 0.5 | RMS of injected per-exposure offsets (mag) |
| `baseline_mag` | −26.53 | Reference magnitude for throughput normalization |
| `ndith` | 2 | Number of dither steps per pointing |
