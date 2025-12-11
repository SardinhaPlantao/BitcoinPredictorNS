"""
Módulo de Machine Learning
Engenharia, seleção e validação de features para modelos ML
"""

from .feature_engineer import FeatureEngineer
from .feature_selector import FeatureSelector
from .feature_validator import FeatureValidator

__all__ = ['FeatureEngineer', 'FeatureSelector', 'FeatureValidator']