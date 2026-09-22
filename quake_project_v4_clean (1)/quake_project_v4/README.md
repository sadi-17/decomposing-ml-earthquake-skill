# quake_project_v4

Mainshock discrimination from earthquake catalogs, with calibrated
probabilities, conformal prediction sets and an ETAS null model.

Three catalogs are used: global ComCat above M4 (no spatial constraint),
SCEDC above M2.5 restricted to Southern California (32-37 N, 122-114 W),
and a Himalaya subset of ComCat (20-32 N, 85-98 E). Events are labelled by a windowed
declustering rule (an event is a mainshock unless a larger event falls within
R km and T days either side), gradient-boosting models are trained on a
chronological split, and the resulting probabilities are recalibrated and
turned into conformal prediction sets and a three-state decision rule. The
sensitivity of every result to the labelling window, to an independent
nearest-neighbour declustering, to completeness, and to a simulated ETAS null
is evaluated alongside the headline metrics.

## Layout

```
quake_project_v4/
├── code/
│   ├── utils.py           configuration, paths, provenance, shared statistics
│   ├── make_dataset.py    FDSN download -> clean -> labels -> features
│   ├── models.py          imputation, tuning, training, recalibration
│   ├── etas.py            nearest-neighbour linkage, ETAS fit and simulator
│   ├── analyses.py        the analyses that produce the result tables
│   ├── report.py          figures and the self-contained HTML report
│   ├── run_analysis.py    stage orchestration
│   └── audit_outputs.py   completeness and consistency checks for a run
├── requirements.txt
└── README.md
```

`raw_data/`, `dataset/` and `results/` are created by the scripts. Nothing in
them belongs in version control or in an archive of the source.

The separation is deliberate: `make_dataset.py` never writes into `results/`,
and `run_analysis.py` never writes into `dataset/` or `raw_data/`. The analysis
refuses to start if the feature files do not match the hashes recorded in
`dataset/data_provenance.json`.

## Running it on Colab

Start from an empty project directory. Nothing needs to be copied in from a
previous run.

1. Upload and extract the archive so that the project is at
   `/content/drive/MyDrive/quake_project_v4`, then mount Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/quake_project_v4
   ```

2. Install the dependencies (re-run this after any runtime restart):

   ```
   !pip install -r requirements.txt
   ```

3. Build the dataset. This downloads both catalogs from FDSN, which is the
   slowest part of the first run; year chunks are cached under
   `raw_data/chunks/`, so an interrupted download resumes.

   ```
   !python code/make_dataset.py --root /content/drive/MyDrive/quake_project_v4
   ```

4. Run the analysis on a T4 GPU:

   ```
   !python code/run_analysis.py --root /content/drive/MyDrive/quake_project_v4 \
       --run-name v4_fresh --gpu
   ```

5. Audit the run:

   ```
   !python code/audit_outputs.py --root /content/drive/MyDrive/quake_project_v4 \
       --run-name v4_fresh
   ```

`--gpu` sends the gradient-boosting fits to the GPU where the library supports
it (CatBoost `task_type=GPU`, XGBoost `device=cuda`). Everything else — the
neighbour searches, the ETAS simulation, the bootstrap, pandas and the
preprocessing — is CPU work and stays there. Without `--gpu` the whole run is
on CPU and is slower but otherwise identical.

## Outputs

Everything from one run lands in `results/<run-name>/`:

```
tables/        one CSV per analysis
figures/       one PNG per figure
models/        fitted models, split indices, raw and recalibrated probabilities
ckpt/          intermediate arrays, so an interrupted run resumes
logs/          console transcript
run_manifest.json
ALL_RESULTS.html
```

`ALL_RESULTS.html` embeds every table and figure in a single file.
`run_manifest.json` records the seed, the pinned catalog end date, package
versions, the SHA-256 of every source file, the dataset provenance and the
fitted ETAS parameters.

A new `--run-name` starts from an empty run folder, so results from different
runs cannot mix. Within a run the checkpoints under `ckpt/` let an interrupted
analysis resume; `--wipe` clears the run folder to force everything to be
recomputed.

## Reproducibility

The global seed is 42 and the catalog end date is pinned to 2026-09-01, so two
runs query the same catalog and see the same splits. GPU training is not
bit-identical across machines; a CPU run is.

## Data

Global and Himalaya catalogs: U.S. Geological Survey ANSS ComCat.
Southern California: SCEDC / Caltech-USGS SCSN, doi:10.7909/C3WD3xH1.
The SCEDC event service returns its full event table, including regional and
teleseismic entries, so that query carries an explicit bounding box; the
Himalaya catalog is a spatial subset of the cleaned global catalog. The bounds
of all three are recorded in `dataset/data_provenance.json` alongside the
magnitude conversions, and a catalog already on disk is re-downloaded if the
query recorded beside it differs from the one configured.
