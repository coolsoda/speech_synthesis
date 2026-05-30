# Checkpoints

This directory is used to store trained model checkpoints and pretrained vocoder checkpoints.


Expected structure:

```text
checkpoints/
├── lflow_alstdi/
│   ├── best.pt
│   └── epoch_XXXX.pt
│
├── lflow_parkceleb/
│   ├── best.pt
│   └── epoch_XXXX.pt
│
├── vocoder/
│   └── hifigan.pt
│
└── baselines/
    ├── omnivoice/
    ├── qwen3_tts/
    ├── voxcpm2/
    ├── xtts_v2/
    ├── ttds/
    └── scp_tts/
```

## L-Flow Checkpoints

After training, L-Flow checkpoints will be saved according to the path specified in `configs/train.yaml`:

```yaml
experiment:
  checkpoint_dir: "checkpoints/lflow_alstdi"
```

The best validation checkpoint is saved as:

```text
best.pt
```

## Vocoder Checkpoint

For waveform generation during inference, place a compatible vocoder checkpoint under:

```text
checkpoints/vocoder/
```

For example:

```text
checkpoints/vocoder/hifigan.pt
```

The path should match the setting in `configs/inference.yaml`:

```yaml
checkpoint:
  vocoder_path: "checkpoints/vocoder/hifigan.pt"
```

## Baseline Checkpoints

External TTS baseline checkpoints should be placed under:

```text
checkpoints/baselines/
```
