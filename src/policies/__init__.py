"""Offloading policies: decide which layer(s) a ready task may run on."""
from .base import AllLayersPolicy, SingleLayerPolicy
from .baseline import TwoWayFogCloudPolicy
from .fuzzy_policy import BALANCED, FuzzyOffloadingPolicy, InputNoise, InputScales, IT2OffloadingPolicy, observe

__all__ = ["AllLayersPolicy", "SingleLayerPolicy", "TwoWayFogCloudPolicy",
           "FuzzyOffloadingPolicy", "IT2OffloadingPolicy", "InputNoise", "InputScales", "observe", "BALANCED"]
