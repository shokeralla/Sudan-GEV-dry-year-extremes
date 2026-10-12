# Censored versus two-part GEV models for annual rainfall maxima in hyper-arid regions: Sudan, 1981–2024

Data, code and outputs for the manuscript *"Censored versus Two-Part GEV Models for Annual Rainfall Maxima in Hyper-Arid Regions: Evidence from Sudan"* (Alshaikh A. Shokeralla).

## Data (`data/`)

| File | Content | Source |
|---|---|---|
| `NASA_POWER_Sudan_points_daily_clean_1981_2024.csv` | Daily precipitation `PRECTOTCORR` (MERRA-2 bias-corrected), ten grid points, 1981–2024 | [NASA POWER Daily API](https://power.larc.nasa.gov/) |
| `NASA_POWER_Sudan_points_EVT_annual_maxima_1981_2024.csv` | Annual maxima (cross-check only) | derived |
| `CHIRPS_Sudan_10pts_daily_1981_2024.csv` | Daily CHIRPS v2.0 precipitation at the same ten points | [Climate Hazards Center](https://www.chc.ucsb.edu/data/chirps) |
| `ne_50m_admin_0_countries.geojson` | Country boundaries (map only) | Natural Earth (public domain) |

Grid points: 10°N 25°E, 10°N 35°E, 12°N 30°E, 15°N 25°E, 15°N 30°E, 15°N 35°E, 18°N 30°E, 20°N 25°E, 20°N 30°E, 20°N 35°E.

Notes
* The NASA POWER daily file contains 16,071 duplicated rows at 15°N 30°E (two merged downloads); all scripts remove them.
* The original NASA POWER API response header (the `-BEGIN HEADER-` block defining `PRECTOTCORR`) will be added to `data/`.

## Code (`code/`)

| Script | Purpose |
|---|---|
| `evt.py` | Model library: standard, censored, two-part (TP-C, TP-T) GEV likelihoods, return levels, profile intervals, PIT diagnostics, simulation |
| `emp.py`, `emp2.py`, `emp3.py` | Empirical analysis: four formulations, LR test of p₀ = G(c) (χ² and bootstrap), full-model goodness of fit, homogeneity, record events |
| `tpc_mixture.py` | Coherent maximum-likelihood counterpart of the conventional TP-C procedure (TP-T restricted to p₀ ≥ G(c)) |
| `trends.py`, `break2015.py`, `nonstat.py` | Secondary analysis: trends, documented 2015 break, non-stationary GEV |
| `chirps.py` | Independent check with CHIRPS |
| `mc.py`, `mc_profile.py`, `mc_optcheck.py`, `mc_summary.py` | Monte Carlo experiment (84 design cells) |
| `figures.py`, `figures_mc.py`, `figures_chirps.py` | Figures 1–8 and S1 |

## Reproduce

```bash
pip install -r requirements.txt
python run_all.py            # full run (Monte Carlo takes several hours on 2 cores); writes out/ and fig/
python run_all.py --quick    # reduced run; writes out_quick/ and fig_quick/ only
```

`out/` holds every table used in the manuscript and `fig/` every figure (300 dpi). Random seeds are fixed in each script.

## Licence

Code: MIT licence. Derived data and outputs: CC BY 4.0. Original NASA POWER and CHIRPS data remain subject to their providers' terms; please cite them as given in the manuscript.

## Citation

See `CITATION.cff`.
