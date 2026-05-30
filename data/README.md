# Data

This repository does not include the complete raw speech datasets used in this study. Due to dataset licensing, privacy restrictions, and data-sharing policies, users should request access to the original datasets from the official sources listed below.

## Dataset Access

### ALS-TDI

The ALS-TDI dataset contains longitudinal speech recordings collected for studying ALS progression. In this project, we use speech recordings, timestamps, transcripts, and ALSFRS-R speech subscores to construct subject-level longitudinal trajectories.

Access request link:

```text
[https://www.als.net/about-als-tdi/contact-als-tdi/]
```

### ParkCeleb

The ParkCeleb dataset contains longitudinal spontaneous speech samples from public figures for studying Parkinson’s disease progression. In this project, we use speech recordings, timestamps, transcripts, and binary PD labels to construct subject-level longitudinal trajectories.

Access request link:

```text
[https://www.nature.com/articles/s41531-024-00817-9#data-availability]
```

## Expected Local Data Organization

After obtaining access to the datasets, please organize the files locally as follows:

```text
data/
├── alstdi/
│   └── audio/
│       ├── ...
│
├── parkceleb/
│   └── audio/
│       ├── ...
│
├── metadata/
│   ├── alstdi_metadata.csv
│   └── parkceleb_metadata.csv
│
└── splits/
    ├── alstdi_split.json
    └── parkceleb_split.json
```

## Metadata Format

Each metadata file should contain one row per recording session. The required columns are:

```text
subject_id,audio_path,transcript,timestamp,label
```

Column descriptions:

| Column       | Description                                                                                         |
| ------------ | --------------------------------------------------------------------------------------------------- |
| `subject_id` | Unique subject or participant identifier.                                                           |
| `audio_path` | Path to the audio file relative to the repository root.                                             |
| `transcript` | Transcript or text prompt associated with the recording.                                            |
| `timestamp`  | Recording date or time point used to define the longitudinal order.                                 |
| `label`      | Normalized clinical progression label in `[0, 1]`, where larger values indicate greater impairment. |

## Dataset Splits

We use subject-level train/test splits to avoid leakage across repeated recordings from the same individual. Split files should be stored under `data/splits/` using the following format:

```json
{
  "train": ["subject_001", "subject_002"],
  "test": ["subject_101", "subject_102"]
}
```

Synthetic samples should only be generated for subjects in the training split. Test subjects should be held out and used only for downstream evaluation.
