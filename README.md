# Mixture-of-Accent-Adapters for Robust ASR: Injecting Accent Cues into Pretrained Whisper

![Image](https://github.com/user-attachments/assets/d36214a7-365f-42aa-9944-cac30f41fbac)



---
## Overview

**MoAA** injects accent cues into a pretrained Whisper model via a bank of lightweight accent adapters and a routing mechanism, conditioned on a probabilistic accent codebook that provides soft (mixture) assignments over accent prototypes to guide adapter selection and combination.

**DHF** is a post-processing module that operates on ASR hypotheses (optionally with references) to filter hallucinations.



---
## Repository contents


- `moaa.py` — MoAA training/evaluation runner built on Whisper-small.
- `dhf.py` — DHF post-processing module operating on prediction CSVs.
- `job.slrm` — SLURM script for running MoAA + DHF.



---
## Environment setup

#### 1) Create and activate an environment

```bash
conda create -n moaa python=3.10 -y
conda activate moaa
```

#### 2) Install dependencies

```bash
pip install -U pip
pip install torch torchvision torchaudio
pip install transformers datasets evaluate numpy pandas safetensors
pip install jiwer
```



---
## Training MoAA


#### 1) Without optional external evaluation sets
```bash
python moaa.py \
  --aesrc_train_dir /path/to/aesrc/train_shards \
  --librispeech_dir /path/to/librispeech \
  --output_root runs/moaa_exp
```

#### 2) With optional external evaluation sets
```bash
python moaa.py \
  --aesrc_train_dir /path/to/aesrc/train_shards \
  --librispeech_dir /path/to/librispeech \
  --aesrc_test_dir /path/to/aesrc/test \
  --openslr_test_dir /path/to/openslr/test \
  --edacc_test_dir /path/to/edacc/test \
  --globe_test_dir /path/to/globe/test \
  --output_root runs/moaa_exp
```



---
## DHF post-processing


#### 1) Reference-free filtering (no ground-truth required)
```bash
python dhf.py \
  --csv_path /path/to/predictions.csv \
  --pred_col pred_text \
  --out_dir dhf_outputs/ref_free_run \
  --no_insertion_analysis \
  --no_eval
```

#### 2) Optional: reference-based analysis
```bash
python dhf.py \
  --csv_path /path/to/predictions.csv \
  --pred_col pred_text \
  --ref_col ref_text \
  --out_dir dhf_outputs/ref_based_run
```



---
## Train MoAA+DHF


```bash
sbatch job.slrm
```






