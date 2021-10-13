"""
Class I MHC processing predictor
"""

from .class1_processing_predictor import Class1ProcessingPredictor
from .class1_processing_neural_network import Class1ProcessingNeuralNetwork

from .version import __version__

__all__ = [
    "__version__",
    "Class1ProcessingPredictor",
    "Class1ProcessingNeuralNetwork",
]
