# Data Science and Cross-Domain Evaluation

Run these notebooks in order. The clinical model is selected using only the fixed clinical validation split. The official
ISIC 2019 test partition is used once as a zero-shot external domain-shift evaluation set.

1. `00_experimentSetup.ipynb`
2. `01_trainClinical.ipynb`
3. `02_evaluateClinical.ipynb`
4. `03_evaluateDomainShift.ipynb`
5. `04_evaluationSummary.ipynb`
6. `05_aggregateSeeds.ipynb`

## Reviewer extension experiments

The original MobileNetV3-Small baseline remains unchanged. The following notebooks add the reviewer-requested controlled
experiments:

7. `06_domainAndLeakageAudit.ipynb` creates a fixed lesion/exact-duplicate-group-aware ISIC development split, measures
   pretrained feature-distribution separation, and exports cross-partition perceptual-match candidates for review.
8. `07_runReviewerExperiment.ipynb` runs one experiment and seed selected by environment variables:
   - `efficientnet`: clinical-trained EfficientNet-B0 replication;
   - `reverse`: ISIC-trained MobileNetV3-Small evaluated on clinical images; or
   - `coral`: clinical-trained MobileNetV3-Small with unsupervised Deep CORAL alignment to ISIC development images.
9. `08_reviewerExperimentSummary.ipynb` validates and aggregates all experiment/seed artifacts.

CORAL uses ISIC development images without their labels. Its checkpoint is still selected with clinical validation
macro-F1. Reverse transfer uses a fixed 17,807/4,408 ISIC development train/validation split and never uses the clinical
test set for selection. All external test sets remain fixed.

Run the full extension suite from the project root:

```powershell
$env:SKIN_RUN_MODE = 'full'
$env:SKIN_SEEDS = '42,67,2026'
$env:SKIN_BATCH_SIZE = '32'
$env:SKIN_EVAL_BATCH_SIZE = '64'
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 'notebooks/ds/06_domainAndLeakageAudit.ipynb'
if ($LASTEXITCODE -ne 0) { throw 'Domain audit failed.' }
foreach ($experimentName in @('efficientnet', 'reverse', 'coral')) {
    $env:SKIN_EXTENSION_EXPERIMENT = $experimentName
    foreach ($randomSeed in @(42, 67, 2026)) {
        $env:SKIN_RANDOM_SEED = "$randomSeed"
        python -m nbconvert --to notebook --execute --inplace `
            --ExecutePreprocessor.timeout=21600 'notebooks/ds/07_runReviewerExperiment.ipynb'
        if ($LASTEXITCODE -ne 0) { throw "Extension failed: $experimentName, seed $randomSeed" }
    }
}
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 'notebooks/ds/08_reviewerExperimentSummary.ipynb'
if ($LASTEXITCODE -ne 0) { throw 'Extension aggregation failed.' }
```

The reverse experiment is substantially slower than the original clinical baseline because it trains on 17,807 images;
the complete three-experiment, three-seed suite may require an overnight run. Run it on AC power with adequate cooling.
Outputs are isolated under `artifacts/extensions/full/` and
`reports/figures/extensions/full/`.

## Smoke verification

From the project root in PowerShell:

```powershell
$env:SKIN_RUN_MODE = 'smoke'
python -m jupyterlab
```

Smoke outputs are written under seed-specific directories such as `artifacts/ds/smoke/seed42/`. They are not paper results.

## Full experiment

On the RTX 3070 Laptop GPU, run:

```powershell
$env:SKIN_RUN_MODE = 'full'
$env:SKIN_SEEDS = '42,67,2026'
$env:SKIN_BATCH_SIZE = '32'
$env:SKIN_EVAL_BATCH_SIZE = '64'
python -m jupyterlab
```

For each seed, set `SKIN_RANDOM_SEED` and execute notebooks 00 through 04. After all seeds complete, execute
`05_aggregateSeeds.ipynb`. Training defaults to 3 classifier-warmup epochs plus up to 12 full fine-tuning
epochs with early stopping patience 4. If CUDA memory is insufficient, set `SKIN_BATCH_SIZE` to `16`. Full outputs are
written under seed-specific directories in `artifacts/ds/full/` and `reports/figures/ds/full/`. The aggregate artifacts are
`artifacts/ds/full/multiSeedSummary.json`, `artifacts/ds/full/multiSeedMetrics.csv`, and `reports/dsEvaluationReport.md`.

For a non-interactive sequential run from the project root:

```powershell
$env:SKIN_RUN_MODE = 'full'
$env:SKIN_SEEDS = '42,67,2026'
$env:SKIN_BATCH_SIZE = '32'
$env:SKIN_EVAL_BATCH_SIZE = '64'
$seedNotebookNames = @(
    '00_experimentSetup.ipynb',
    '01_trainClinical.ipynb',
    '02_evaluateClinical.ipynb',
    '03_evaluateDomainShift.ipynb',
    '04_evaluationSummary.ipynb'
)
foreach ($randomSeed in @(42, 67, 2026)) {
    $env:SKIN_RANDOM_SEED = "$randomSeed"
    foreach ($notebookName in $seedNotebookNames) {
        python -m nbconvert --to notebook --execute --inplace `
            --ExecutePreprocessor.timeout=7200 "notebooks/ds/$notebookName"
        if ($LASTEXITCODE -ne 0) { throw "Notebook failed: seed $randomSeed, $notebookName" }
    }
}
python -m nbconvert --to notebook --execute --inplace `
    --ExecutePreprocessor.timeout=7200 'notebooks/ds/05_aggregateSeeds.ipynb'
if ($LASTEXITCODE -ne 0) { throw 'Seed aggregation failed.' }
```

Reported metrics are macro-F1, balanced accuracy, per-class recall, and confusion matrices. Macro-F1 selects the
checkpoint. Both scalar test metrics include stratified bootstrap 95% confidence intervals per seed. The paper-facing
multi-seed report uses the mean and sample standard deviation across the three independently trained seeds.

The official ISIC 2019 data are distributed under CC-BY-NC. Follow the attribution requirements documented by the
official ISIC Challenge dataset page when preparing the paper.
