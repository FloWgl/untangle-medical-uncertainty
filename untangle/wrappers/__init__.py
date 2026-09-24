"""Implementations of supported uncertainty wrappers."""

__all__ = [
    "BaseCorrectnessPredictionWrapper",
    "BaseLossPredictionWrapper",
    "CEBaselineWrapper",
    "CorrectnessPredictionWrapper",
    "DDUWrapper",
    "DUQWrapper",
    "DeepCorrectnessPredictionWrapper",
    "DeepEnsembleWrapper",
    "DeepLossPredictionWrapper",
    "DirichletWrapper",
    "DistributionalWrapper",
    "EDLWrapper",
    "FastDeepEnsembleWrapper",
    "HETXLWrapper",
    "HetClassNNWrapper",
    "LaplaceWrapper",
    "LossPredictionWrapper",
    "MCDropoutWrapper",
    "MahalanobisWrapper",
    "ModelWrapper",
    "PostNetWrapper",
    "SNGPWrapper",
    "SWAGWrapper",
    "ShallowEnsembleWrapper",
    "SpecialWrapper",
    "TemperatureWrapper",
]


class _MissingWrapper:
    pass
from .ce_baseline_wrapper import CEBaselineWrapper
from .correctness_prediction_wrapper import (
    BaseCorrectnessPredictionWrapper,
    CorrectnessPredictionWrapper,
    DeepCorrectnessPredictionWrapper,
)
from .deep_ensemble_wrapper import DeepEnsembleWrapper
from .loss_prediction_wrapper import (
    BaseLossPredictionWrapper,
    DeepLossPredictionWrapper,
    LossPredictionWrapper,
)
from .model_wrapper import (
    DirichletWrapper,
    DistributionalWrapper,
    ModelWrapper,
    SpecialWrapper,
)
from .shallow_ensemble_wrapper import ShallowEnsembleWrapper


# Helper to optionally import a wrapper and fall back to _MissingWrapper on error
def _optional_import(module_name: str, symbol_name: str):
    try:
        module = __import__(f"untangle.wrappers.{module_name}", fromlist=[symbol_name])
        return getattr(module, symbol_name)
    except Exception:
        return _MissingWrapper


# Optional wrappers (may require extra dependencies). Import once centrally.
DDUWrapper = _optional_import("ddu_wrapper", "DDUWrapper")
DUQWrapper = _optional_import("duq_wrapper", "DUQWrapper")
EDLWrapper = _optional_import("edl_wrapper", "EDLWrapper")
FastDeepEnsembleWrapper = _optional_import("fast_deep_ensemble_wrapper", "FastDeepEnsembleWrapper")
HetClassNNWrapper = _optional_import("hetclassnn_wrapper", "HetClassNNWrapper")
HETXLWrapper = _optional_import("hetxl_wrapper", "HETXLWrapper")
LaplaceWrapper = _optional_import("laplace_wrapper", "LaplaceWrapper")
MahalanobisWrapper = _optional_import("mahalanobis_wrapper", "MahalanobisWrapper")
MCDropoutWrapper = _optional_import("mc_dropout_wrapper", "MCDropoutWrapper")
PostNetWrapper = _optional_import("postnet_wrapper", "PostNetWrapper")
SNGPWrapper = _optional_import("sngp_wrapper", "SNGPWrapper")
SWAGWrapper = _optional_import("swag_wrapper", "SWAGWrapper")
TemperatureWrapper = _optional_import("temperature_wrapper", "TemperatureWrapper")
