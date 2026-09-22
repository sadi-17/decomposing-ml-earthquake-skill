# Decomposing Apparent Skill in Machine-Learning Event Classification

## A Multi-Catalog Study of Label Structure, Feature Coupling, and Operational Generalization


## Overview

This repository contains the complete reproducible machine learning pipeline developed for the study:

**"Decomposing Apparent Skill in Machine-Learning Event Classification: A Multi-Catalog Study of Label Structure, Feature Coupling, and Operational Generalization"**

The project investigates why machine-learning models can achieve very high performance when trained and evaluated on retrospectively constructed scientific labels.

Using earthquake event-role classification as a case study, this work evaluates whether apparent machine-learning skill originates from:

- target construction rules,
- feature–label coupling,
- magnitude-derived structural information,
- retrospective evaluation design,
- or transferable predictive information.

Rather than proposing a forecasting system, this study develops a diagnostic framework for evaluating scientific machine-learning problems where labels are constructed using future information.

---

# Research Objectives

This repository addresses the following research questions:

1. How much apparent machine-learning performance is produced by retrospectively constructed labels?

2. How much discrimination can be obtained from simple target-aligned structural features?

3. Does high held-out performance remain when evaluation is restricted to events whose future role is unresolved?

4. Can synthetic clustered catalogs reproduce apparent machine-learning performance without explicit parent–offspring magnitude relationships?

5. How robust are conclusions under:
   - alternative target definitions,
   - feature ablation,
   - calibration analysis,
   - uncertainty quantification,
   - and cross-region transfer?


---

# Dataset Description

The analysis uses three earthquake catalogs.

| Dataset | Source | Region | Number of Events |
|---|---|---|---|
| Global ComCat | USGS ANSS Comprehensive Catalog | Worldwide | 545,220 |
| SCEDC | Southern California Earthquake Data Center | Southern California | 45,648 |
| Himalaya ComCat | USGS ComCat regional subset | Himalaya | 4,710 |


Dataset construction is fully automated.

The dataset generation pipeline records:

- data source
- filtering criteria
- spatial boundaries
- temporal range
- event counts
- feature availability
- provenance metadata
- file hashes


---

# Machine Learning Models

The framework evaluates three gradient-boosted tree models:

- CatBoost
- XGBoost
- LightGBM

Training: 60%
Validation: 10%
Calibration: 15%
Testing: 15%



Hyperparameter optimization is performed using Optuna.

---

# Analysis Framework

## 1. Retrospective Target Construction

Binary event-role labels are generated using space–time–magnitude neighborhood rules.

Reference configuration:

Spatial radius: 50 km
Temporal window: 30 days



Alternative target definitions are evaluated to measure sensitivity to label construction.


---

## 2. Feature–Label Coupling Analysis

The study evaluates:

- magnitude difference features
- magnitude-only baselines
- structural feature importance
- mutual information between features and labels
- complete magnitude-family ablation


The goal is to determine whether apparent model skill depends strongly on information embedded in the target definition itself.


---

## 3. Operationally Conditioned Evaluation

Conventional held-out evaluation is compared with a more restrictive subset:
Backward local maxima


These are events whose future role cannot already be determined from larger previous events within the defined neighborhood.

This experiment evaluates whether pooled retrospective performance persists under a more difficult information state.


---

## 4. ETAS Procedural Null Analysis

Synthetic catalogs are generated using:

Epidemic-Type Aftershock Sequence (ETAS)


The complete machine-learning pipeline is applied to synthetic catalogs.

The purpose is not to reproduce earthquakes perfectly, but to estimate how much apparent discrimination can emerge from:

- clustered event structure,
- retrospective labels,
- and evaluation design.


---

## 5. Calibration and Uncertainty Quantification

The framework evaluates probability reliability using:

- Expected Calibration Error (ECE)
- Brier Score
- Negative Log Likelihood (NLL)


Calibration methods include:

- Platt scaling
- Isotonic regression
- Temperature scaling


Prediction uncertainty is evaluated using conformal prediction methods.


---

## 6. Cross-Catalog Distribution Shift

Transfer experiments evaluate geographic generalization:


decomposing-ml-earthquake-skill/

│
├── code/
│ ├── make_dataset.py
│ ├── analyses.py
│ ├── models.py
│ ├── etas.py
│ ├── run_analysis.py
│ ├── audit_outputs.py
│ ├── report.py
│ └── utils.py
│
├── configs/
│
├── data/
│ └── provenance/
│
├── results/
│ ├── figures/
│ ├── tables/
│ └── manifests/
│
├── manuscript/
│
├── requirements.txt
├── environment.yml
├── CITATION.cff
└── README.md



---


**Citation**

If you use this repository, please cite:

Sadi TH, Ruthbah CA.

Decomposing Apparent Skill in Machine-Learning Event Classification:
A Multi-Catalog Study of Label Structure, Feature Coupling, and Operational Generalization.

Advances in Distributed Computing and Artificial Intelligence Journal.



**Authors
Talim Hossain Sadi

Department of Computer Science and Engineering
BRAC University

Chowdhury Aseer Ruthbah

Department of Computer Science and Engineering
BRAC University**


Models are trained using chronological data splitting:

