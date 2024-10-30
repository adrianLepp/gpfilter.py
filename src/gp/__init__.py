"""examples for the implemented gp filter.
"""

from .kernel import ConvolvedProcessKernel
from .multi_gp import BatchIndependentMultitaskGPModel, MultitaskGPModel, ConvolvedGPModel

__all__ = [
    "ConvolvedProcessKernel",
    "BatchIndependentMultitaskGPModel",
    "MultitaskGPModel",
    "ConvolvedGPModel",
]