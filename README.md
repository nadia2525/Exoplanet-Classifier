# Exoplanet Classification: Confirmed vs. False Positive

A machine learning project analyzing NASA Kepler mission data to classify
candidate exoplanets as confirmed planets or false positives, based on
transit and stellar characteristics.

## Why this project

Kepler found thousands of "candidates," but not everything that looks like
a planet transit actually is one (eclipsing binaries, instrumental noise,
and other stars in the same pixel can mimic a transit signal). This project
builds a model to separate real planets from false positives, and asks:
**which physical properties are most predictive of a real planet?**

## Data source

NASA Exoplanet Archive — Kepler Objects of Interest (Cumulative table).

Direct CSV download (no login needed):
https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv

Save it to `data/cumulative.csv`.

Key columns you'll use:
- `koi_disposition` — target label (CONFIRMED / FALSE POSITIVE / CANDIDATE)
- `koi_period` — orbital period (days)
- `koi_duration` — transit duration (hours)
- `koi_depth` — transit depth (ppm)
- `koi_prad` — planetary radius (Earth radii)
- `koi_teq` — equilibrium temperature (K)
- `koi_insol` — insolation flux
- `koi_model_snr` — transit signal-to-noise ratio
- `koi_steff`, `koi_slogg`, `koi_srad` — stellar temperature, gravity, radius

## Project structure

```
exoplanet-project/
├── data/                 # put cumulative.csv here
├── 01_load_clean.py      # load + clean the raw data
├── 02_eda.py             # exploratory analysis + plots
├── 03_model.py           # train + evaluate classifiers
├── app.py                # Streamlit interactive dashboard
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
python 01_load_clean.py
python 02_eda.py
python 03_model.py
streamlit run app.py
```

## Results

*(Fill this in once you've run the pipeline — e.g. best model, accuracy/
ROC-AUC, top 3 predictive features, and 2-3 sentences on what that tells
you physically about how false positives differ from real planets.)*

## Live demo

[Try the live app here](https://exoplanet-classifier-zleqffftafd8prv7lsfeng.streamlit.app)
