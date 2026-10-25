# Project 2 datasets

These are actual local copies of the datasets used in the experiments, not links to Project 1.

- PathMNIST: `pathmnist.npz` (preserves the official train/validation/test arrays).
- PBMC3K: `pbmc3k/filtered_gene_bc_matrices/hg19/` (10x count matrix and accompanying gene/barcode files).

`dataset_manifest.json` records each source path, size and SHA-256 checksum. All copied files were verified against their originals. Project 1 files were not changed. These files retain their original dataset terms; copying does not change ownership or licensing.

Run commands from the Project 2 directory using `data/pathmnist.npz` or `data/pbmc3k/filtered_gene_bc_matrices/hg19`. Previously saved experiment plans and provenance retain their original source paths to preserve the historical record. For a new experiment based on an archived configuration, copy the configuration to a new experiment directory, update its dataset paths to these local copies, and choose fresh output directories. Copying the files does not perform a final-test evaluation.

## Setup after cloning

The raw datasets are not stored in Git. From the repository root, run `python3 scripts/setup_datasets.py` to download the official files and verify them against `data/checksums.json`. Run `python3 scripts/setup_datasets.py --verify-only` to check existing copies without network access. The script preserves publisher splits, extracts only the three required 10x files, and refuses to overwrite mismatching data. Dataset download is setup; each modeling workflow then runs with one command.
