# Exoplanet Classification: Confirmed vs. False Positive

A machine learning project analyzing NASA Kepler mission data to classify
candidate exoplanets as confirmed planets or false positives, based on
transit and stellar characteristics.

## Why this project

Kepler found thousands of "candidates," but not everything that looks like
a planet transit actually is one (eclipsing binaries, instrumental noise,
and other stars in the same pixel can mimic a transit signal). This project
builds a model to separate real planets from false positives, and asks:
which physical properties are most predictive of a real planet?

## Data source

NASA Exoplanet Archive — Kepler Objects of Interest (Cumulative table).

Direct CSV download (no login needed):
https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv

Saved it to `data/cumulative.csv`.

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

Two machine learning models were evaluated for classifying Kepler objects as **Confirmed exoplanets** or false positives.

### Logistic Regression

The Logistic Regression model achieved an overall accuracy of 81% and a ROC-AUC score of 0.881. It performed reasonably well as a baseline model, with an F1-score of 0.76 for confirmed exoplanets.

### Random Forest

The Random Forest model achieved the best overall performance, with an accuracy of 91% and a ROC-AUC score of 0.971. It achieved an F1-score of 0.88 when identifying confirmed exoplanets, outperforming the Logistic Regression model.

### Feature Importance

The Random Forest model identified the following features as particularly important for classification:

1. Planetary radius (`koi_prad`) — 0.225
2. Transit signal-to-noise ratio (`koi_model_snr`) — 0.167
3. Orbital period (`koi_period`) — 0.108
4. Transit duration (`koi_duration`) — 0.101
5. Transit depth (`koi_depth`) — 0.092

Overall, the results indicate that the Random Forest model is substantially more effective than Logistic Regression for this classification task. The importance of features related to planetary characteristics and transit signals suggests that these measurements play a significant role in distinguishing confirmed exoplanets from false positives.


## Live demo

[Try the live app here](https://exoplanet-classifier-zleqffftafd8prv7lsfeng.streamlit.app)
