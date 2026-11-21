# GastroVL

Code and derived data for **GastroVL**, an open dataset of 10,000 white-light
esophagogastroduodenoscopy (EGD) images, each paired with a structured
free-text description written by experienced endoscopists.

This repository contains the scripts that produce the dataset-level statistics,
the duplicate screening, the inter-rater agreement analysis and Figures 8 and 9
of the Data Descriptor. The images and descriptions themselves are not stored
here. They are deposited on Zenodo and released under CC0.

## The dataset

| | |
|---|---|
| Records | 10,000 image–text pairs |
| Patients | 648, one examination each |
| Anatomical sites | 10 |
| Descriptions | location, morphology, size, margin, color |
| Licence | CC0 1.0 Universal |
| Repository | Zenodo, https://doi.org/10.5281/zenodo.20783082 |

The release is organised as ten anatomical-site folders. Inside each folder,
two subfolders carry a three-digit name in which the first two digits repeat
the site code and the third gives the clinical class (0 = normal,
1 = abnormal). Each of these holds an `image` folder with PNG files and a
`text` folder with TXT files of the same base name. `index.csv` in the root
lists `image_id`, `patient_id`, `anatomical_site`, `class`, `image_path` and
`text_path`.

## What is in this repository

```
scripts/
  verify_duplicates.py           MD5, pHash and SSIM screening, one pass
  agreement_statistics.py        agreement and audit statistics, bootstrap CIs
  image_statistics.py            width, height and file size of the released images
  figure8_confusion_heatmaps.py  Figure 8, anatomical-site discrepancies
  figure9_site_consistency.py    Figure 9, site-specific consistency
baseline/
  prepare_gpu_bundle.py          builds the reduced bundle uploaded to the GPU host
  train_lora.py                  LoRA fine-tuning of Qwen2.5-VL-7B-Instruct
  run_baseline.py                greedy generation, zero-shot or with an adapter
  evaluate.py                    BLEU, ROUGE-L and attribute F1, with strata
data/
  confusion_matrix.csv           site-by-site classification matrix, 10 x 10
  site_agreement.csv             images and discordant images per site
figures/                         output directory
```

Paths to the dataset are read from the environment, so nothing needs editing
before a run:

| Variable | Used by | Default |
|---|---|---|
| `GASTROVL_ROOT` | duplicate screening, bundle preparation | `GastroVL` |
| `GASTROVL_OUT` | duplicate screening | `outputs` |
| `GASTROVL_BASE` | agreement statistics, bundle preparation | `.` |
| `GASTROVL_SPLIT` | bundle preparation | `outputs/official_split.csv` |
| `GASTROVL_MODEL` | training and inference | local Qwen2.5-VL-7B-Instruct |
| `GASTROVL_BUNDLE` | training and inference | `train_bundle`, `gpu_bundle` |

`data/` holds the aggregated counts behind Figures 8 and 9, so both figures can
be rebuilt without access to the images.


## Quick start

```bash
git clone https://github.com/hwnzl/GastroVL.git
cd GastroVL
pip install -r requirements.txt
```

Rebuild the two figures from the aggregated counts:

```bash
python scripts/figure8_confusion_heatmaps.py --csv data/confusion_matrix.csv --out figures/Figure8
python scripts/figure9_site_consistency.py --csv data/site_agreement.csv --out figures/Figure9
```

Both scripts write PNG, PDF and TIFF at 600 dpi and 183 mm width, the full
column width used by the journal.

Run the analyses on a local copy of the dataset:

```bash
export GASTROVL_ROOT=/path/to/GastroVL
python scripts/verify_duplicates.py            # writes outputs/near_duplicates_ssim.csv
python scripts/image_statistics.py --root "$GASTROVL_ROOT"

python scripts/agreement_statistics.py --r1 audit_rater_1.csv --r2 audit_rater_2.csv
```

`verify_duplicates.py` reports the highest SSIM among the candidate pairs and
the number of pairs at or above the threshold. `agreement_statistics.py` reads
the two rating sheets exported from the audit interface, one row per audited
pair, and writes every proportion with its numerator, denominator and
bootstrap interval.

## Baseline evaluation

The partition is drawn by patient, so no examination contributes to two
partitions. Of the 648 patients, 70% are assigned to training, 15% to
validation and 15% to test, which gives 7,025 training pairs and a held-out
test set of 1,472 images from 97 patients.

```bash
pip install -r requirements-baseline.txt
export GASTROVL_ROOT=/path/to/GastroVL

python baseline/prepare_gpu_bundle.py --split test
python baseline/train_lora.py --bundle train_bundle --epochs 1 --out lora_gastrovl
python baseline/run_baseline.py --bundle gpu_bundle --adapter lora_gastrovl --out results_lora.csv
python baseline/run_baseline.py --bundle gpu_bundle --out results_zeroshot.csv
python baseline/evaluate.py --pred results_lora.csv --freq reference_freq.csv
```

Training one epoch takes 878 steps at an effective batch size of 8 and yields
47,589,376 trainable parameters, 0.57% of the model. The adapter weights are
not stored in this repository, because the base model plus the settings above
reproduce them.

## What the scripts reproduce

Run on the released images, the scripts return the values reported in the
Data Descriptor.

- `image_statistics.py` gives widths from 336 to 1,920 px, heights from 299 to
  1,107 px and file sizes from 223 to 3,860 KB.
- `duplicate_screening.py` finds no byte-identical pair, retains about 1,100
  candidate pairs at a Hamming distance of five or less, and returns a highest
  SSIM of 0.849, below the 0.90 threshold, so no image is removed. The exact
  number of candidates depends on the hash implementation; the figures in the
  Data Descriptor were produced with a 64-bit pHash (`imagehash.phash`, hash
  size 8) and a Hamming threshold of five.
- `agreement_statistics.py` gives kappa = 0.818 for anatomical site and
  kappa = 0.851 for the normal/abnormal judgment, and on the 200-pair audit an
  exact agreement of 97.0% with AC1 = 0.969 and a linear-weighted kappa of
  0.387.

Confidence intervals are obtained by resampling images, not patients. Images
from one examination are correlated, so an interval that resamples images is
narrower than one that resamples patients. This is stated in the Data
Descriptor.

## Environment

Python 3.10 or later. Tested with numpy 1.26, pandas 2.2, scikit-image 0.24,
imagehash 4.3, Pillow 10.4 and matplotlib 3.9 on Linux and Windows.

## Citation

If you use the dataset or this code, please cite the Data Descriptor and the
dataset record.

```
Xu, Z. et al. A multimodal dataset of esophagogastroduodenoscopy images with
structured textual annotations. Sci. Data (2026).

Xu, Z. et al. GastroVL: a multimodal dataset of esophagogastroduodenoscopy
images with structured textual annotations. Zenodo
https://doi.org/10.5281/zenodo.20783082 (2026).
```

## Licence

The code in this repository is released under the MIT licence. The dataset is
released under CC0 1.0 Universal.

## Contact

Zhulin Xu, Department of Gastroenterology, the First Affiliated Hospital of
Soochow University. Please open an issue for questions about the code.
