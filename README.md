# L-Flow

This repository contains the implementation for **L-Flow: Longitudinal Flow Matching for Progression-Aware Speech Biomarker Modeling**.


## Repository Structure

```text
speech_synthesis/
├── configs/
├── data/
├── src/
├── checkpoints/
├── outputs/
├── notebooks/
├── requirements.txt
└── README.md
```

## Directory Overview

### `configs/`

Configuration files for training, inference, baseline generation, and downstream evaluation.

```text
configs/
├── train.yaml
├── inference.yaml
├── baselines.yaml
└── downstream.yaml
```

### `data/`

Dataset access instructions.


### `src/models/`

Core L-Flow implementation.

```text
src/models/
├── data.py
├── preprocessing.py
├── progression_encoder.py
├── generator.py
├── losses.py
├── train.py
├── inference.py
└── utils.py
```

### `src/baselines/`

Baseline augmentation methods.

```text
src/baselines/
├── speed_perturb.py
├── vtlp.py
└── tts_baselines.py
```

### `src/evaluation/`

Evaluation scripts for downstream regression, progression consistency, and speech validity.

```text
src/evaluation/
├── downstream_regression.py
├── progression_consistency.py
└── speech_validity.py
```

### `src/scripts/`

Command-line entry points for running the main project workflow.

```text
src/scripts/
├── prepare_data.py
├── train_lflow.py
├── generate_synthetic.py
├── run_baselines.py
└── run_downstream.py
```

### `checkpoints/`

Directory for trained L-Flow checkpoints, vocoder checkpoints, and external baseline model weights.

Model weights are not tracked by Git. See `checkpoints/README.md`.


## Installation

Create a Conda environment:

```bash
conda env create -f environment.yml
conda activate lflow
```

Or install dependencies with pip:

```bash
pip install -r requirements.txt
```





