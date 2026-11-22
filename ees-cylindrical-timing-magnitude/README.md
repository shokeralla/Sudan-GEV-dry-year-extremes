# Joint timing and magnitude of annual maximum daily rainfall in Sudan

Code and data for the manuscript

> Shokeralla AA. *A cylindrical extreme value model for the timing and magnitude of annual maximum rainfall in Sudan.* Submitted to *Environmental and Ecological Statistics*.

The model couples a von Mises distribution for the date of the annual maximum with a GEV distribution for its size, whose location depends on time, on the signed lateness of the event and on its distance from the seasonal peak (models M0–M3). It is applied to CHIRPS v2.0 daily rainfall, 1981–2024, at ten sites in Sudan, with NASA POWER used as a screened comparison.

## Contents

| Path | What it holds |
| --- | --- |
| `chirps/` | CHIRPS v2.0 daily rainfall (0.1° box mean), one CSV per site, downloaded from ClimateSERV |
| `data/power/` | NASA POWER daily `PRECTOTCORR` files for the same ten points |
| `data/indices/` | Monthly Niño 3.4 and DMI series from NOAA PSL |
| `scripts/` | All analysis code (see below) |
| `out/` | Online Resource 1 (`ESM_1.xlsx`), annual maxima and dates (`pairs_chirps.csv`, `pairs_power.csv`), homogeneity tests, model results |
| `fig/` | Figures 1–5 of the manuscript (600 dpi PNG; TIFF copies are written by the scripts) |

## Reproducing the results

Python 3.11 or later with `numpy`, `pandas`, `scipy`, `matplotlib`, `openpyxl`, `statsmodels` and `pymannkendall`. Run from this folder:

```bash
python scripts/chirps.py          # optional: re-download CHIRPS into chirps/
python scripts/extract.py         # NASA POWER annual maxima -> out/pairs_power.csv
python scripts/chirps_pairs.py    # CHIRPS annual maxima + homogeneity tests -> out/pairs_chirps.csv, out/homogeneity.csv
python scripts/models.py out/pairs_chirps.csv chirps 200   # M0-M3 fits, return levels, pooling (about 10 min)
python scripts/extra.py           # descriptive and circular summaries, product comparison
python scripts/watson.py          # bootstrap Watson U2 tests for the von Mises margin
python scripts/figs.py            # Figs 2-5 -> fig/ (PNG and TIFF)
python scripts/map.py             # Fig 1 site map (needs geopandas<1.0, which bundles Natural Earth boundaries)
```

`models.py` uses fixed random seeds, so bootstrap results are reproducible.

## Data sources

- CHIRPS v2.0: Funk et al. (2015), *Scientific Data* 2:150066, https://www.chc.ucsb.edu/data/chirps
- NASA POWER: https://power.larc.nasa.gov
- Niño 3.4 and DMI: NOAA Physical Sciences Laboratory, https://psl.noaa.gov

Note: the NASA POWER annual maxima contain a two- to three-fold break after 2014 (Appendix D of the manuscript) and should not be used for trend analysis.

## Contact

Alshaikh A. Shokeralla, sshokeralla@gmail.com, ORCID 0000-0002-1518-475X
