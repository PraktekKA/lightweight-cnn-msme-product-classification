# Lightweight CNN Architectures for Automatic Product Image Classification

Code, data split, and results for the manuscript:

> Zulkifli, "Lightweight CNN Architectures for Automatic Product Image Classification: A Benchmark Study Toward MSME E-Commerce Cataloging," *Journal of Technology Informatics and Engineering (JTIE)*, under review.

The study benchmarks **MobileNetV3-Small**, **EfficientNet-B0**, and **ShuffleNetV2** (ImageNet-pretrained, fully fine-tuned) on ten product classes of the public *Fashion Product Images (Small)* dataset, with and without data augmentation, each trained with three random seeds.

## Dataset

Images are **not** redistributed here. Download them from Kaggle under the dataset's own terms:
<https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small>

`data/split_dataset.csv` lists every image used (`id` = Kaggle image id), its class (`articleType`), integer label, and subset (`train` / `val` / `test`). The ten classes are the most frequent article types; at most 400 images per class were sampled (`random_state = 42`) and split 70/15/15 with stratification (`random_state = 42`). `data/distribusi_kelas_dataset_asli.csv` gives the class counts of the full source dataset.

## Repository structure

| Path | Content |
|---|---|
| `notebooks/01_initial_experiment.ipynb` | Initial single-seed experiment of the original submission |
| `notebooks/02_revision_experiment_multi_seed.ipynb` | Revision experiment: 3 models × 2 augmentation settings × 3 seeds, bootstrap CIs, McNemar tests, GPU/CPU latency, peak GPU memory, confidence-based verification |
| `scripts/mem_probe.py` | Helper used for the (inconclusive) process-level CPU memory probe |
| `results/experiment_config.json` | Classes, seeds, hyperparameters, hardware, library versions |
| `results/tabel_ringkasan_seed.csv` | Mean ± SD over seeds per model and augmentation setting |
| `results/tabel_per_run.csv` | Per-run metrics with 95% bootstrap CI of accuracy |
| `results/tabel_mcnemar.csv`, `results/tabel_mcnemar_noaug.csv` | Pairwise McNemar tests (with / without augmentation), Holm-adjusted |
| `results/tabel_augmentasi_mcnemar.csv` | With vs. without augmentation per model and seed |
| `results/tabel_latensi.csv`, `results/tabel_memori.csv` | Latency (GPU/CPU, model-only/end-to-end) and memory |
| `results/tabel_confidence.csv` | Coverage and accuracy for confidence thresholds 0.5–0.95 |
| `results/predictions/*.npz` | Test-set labels (`y`) and softmax outputs (`p`) for all 18 runs |
| `results/figures/` | Confusion matrix (3 seeds) and confidence trade-off figure |

## Reproducing

1. Open `notebooks/02_revision_experiment_multi_seed.ipynb` in Google Colab, select a T4 GPU runtime, and run all cells. The notebook downloads the dataset with `kagglehub`, stores per-run results in Google Drive (folder `hasil_revisi_jtie`), and skips completed runs if restarted.
2. Training: Adam (lr 0.001), batch 32, max. 15 epochs, ReduceLROnPlateau (factor 0.5, patience 2), early stopping (patience 4), input 224 × 224, seeds 42, 123, 2026.

GPU computation is not fully deterministic, so individual runs may differ slightly from the reported values; the paper therefore reports means, standard deviations, and paired tests.

Original environment: Google Colab, NVIDIA Tesla T4, Intel Xeon @ 2.00 GHz (2 vCPU), Python 3.13, PyTorch 2.14.

## License

Code is released under the MIT License. The dataset remains subject to its original Kaggle license and terms of use.
