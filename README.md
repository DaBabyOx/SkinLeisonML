# Skin Lesion Cross-Domain Dataset Preparation

Phase 1 prepares clinical and ISIC 2019 datasets for a later cross-domain classification study. It performs discovery,
validation, explicit label harmonization, duplicate review, leakage-aware splitting, and descriptive analysis. It does
not train models or perform domain adaptation.

## Data placement

Raw data is resolved in this order:

1. `SKIN_DATA_ROOT`, when set;
2. `data/raw/`;
3. the existing sibling `../Data/` directory.

Raw files are never modified. Clinical ZIP archives are safely extracted into `data/interim/clinicalImages/`.

Expected source files include the clinical `metadata.csv` and image ZIP archives, ISIC image directories and metadata,
and the official ISIC 2019 training and test ground-truth CSVs. Missing labels are reported as a blocking prerequisite
and are never inferred. The official ISIC test partition is reserved for external domain-shift evaluation.

## Environment

Use Python 3.11 or newer:

```powershell
cd skin-domain-shift
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Execution order

Open JupyterLab from the project root and run each notebook top to bottom:

1. `notebooks/01_datasetDiscovery.ipynb`
2. `notebooks/02_dataValidation.ipynb`
3. `notebooks/03_labelHarmonization.ipynb`
4. `notebooks/04_duplicateDetection.ipynb`
5. `notebooks/05_datasetSplitting.ipynb`
6. `notebooks/06_datasetAnalysis.ipynb`

```powershell
jupyter lab
```

Each notebook reads artifacts saved by the previous stage, uses deterministic seed 42 where randomness is required,
and includes lightweight assertions or synthetic behavior checks. Expensive image hashes are cached by relative path,
file size, and modification time.

## Outputs

- `data/interim/`: discovery, validation, extracted clinical images, and hash cache.
- `data/processed/`: harmonized manifests, duplicate tables, deterministic splits, and statistics.
- `reports/figures/`: publication-quality descriptive figures.
- `reports/datasetReport.md`: Phase 1 status and readiness summary.

Phase 2 must not start until the readiness report confirms that official ISIC labels are present and harmonized data is ready.

## Data science and evaluation

The Phase 2 clinical-training and zero-shot ISIC evaluation notebooks are under `notebooks/ds/`. See
`notebooks/ds/README.md` for smoke and three-seed full execution instructions. Official ISIC test images are reserved
exclusively for external domain-shift evaluation and do not influence checkpoint selection.
