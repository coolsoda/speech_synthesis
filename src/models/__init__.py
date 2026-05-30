"""
- longitudinal speech dataset loading;
- audio/text preprocessing;
- progression-aware longitudinal encoding;
- conditional flow-matching speech generation;
- L-Flow training;
- synthetic speech generation;
- loss functions and utility helpers.
"""

from .data import (
    RecordingSession,
    LongitudinalSubject,
    InferenceQuery,
    LongitudinalSpeechDataset,
    LongitudinalInferenceDataset,
    load_metadata,
    load_split,
    build_subject_trajectories,
    build_interpolation_queries,
    create_dataset_from_config,
)

from .preprocessing import (
    AudioConfig,
    MelConfig,
    TextConfig,
    MaskConfig,
    LFlowPreprocessor,
    LFlowTrainingTransform,
    LFlowInferenceTransform,
    build_preprocessor_from_config,
    lflow_training_collate_fn,
    lflow_inference_collate_fn,
)

from .progression_encoder import (
    LongitudinalProgressionEncoder,
    MetadataOnlyProgressionEncoder,
    build_progression_encoder_from_config,
)

from .generator import (
    ConditionalFlowGenerator,
    LFlowModel,
    build_generator_from_config,
    build_lflow_model_from_config,
)

from .losses import (
    LFlowLoss,
    LFlowLossConfig,
    FlowMatchingLoss,
    ProgressionSmoothnessLoss,
    OrdinalSeverityLoss,
    build_loss_from_config,
)

from .train import (
    train_lflow,
    train_from_config_path,
)

from .inference import (
    generate_synthetic_dataset,
    generate_from_config_path,
)

__version__ = "0.1.0"

__all__ = [
    # Data
    "RecordingSession",
    "LongitudinalSubject",
    "InferenceQuery",
    "LongitudinalSpeechDataset",
    "LongitudinalInferenceDataset",
    "load_metadata",
    "load_split",
    "build_subject_trajectories",
    "build_interpolation_queries",
    "create_dataset_from_config",

    # Preprocessing
    "AudioConfig",
    "MelConfig",
    "TextConfig",
    "MaskConfig",
    "LFlowPreprocessor",
    "LFlowTrainingTransform",
    "LFlowInferenceTransform",
    "build_preprocessor_from_config",
    "lflow_training_collate_fn",
    "lflow_inference_collate_fn",

    # Progression encoder
    "LongitudinalProgressionEncoder",
    "MetadataOnlyProgressionEncoder",
    "build_progression_encoder_from_config",

    # Generator
    "ConditionalFlowGenerator",
    "LFlowModel",
    "build_generator_from_config",
    "build_lflow_model_from_config",

    # Losses
    "LFlowLoss",
    "LFlowLossConfig",
    "FlowMatchingLoss",
    "ProgressionSmoothnessLoss",
    "OrdinalSeverityLoss",
    "build_loss_from_config",

    # Training
    "train_lflow",
    "train_from_config_path",

    # Inference
    "generate_synthetic_dataset",
    "generate_from_config_path",
]