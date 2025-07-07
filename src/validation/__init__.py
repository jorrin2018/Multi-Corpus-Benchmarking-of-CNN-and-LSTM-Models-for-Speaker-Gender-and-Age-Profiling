"""
Módulo de validación científica para Speaker Profiling.

Este módulo proporciona herramientas para validar la reproducibilidad,
exactitud y consistencia científica de los experimentos y resultados.

Componentes principales:
    - ResultValidator: Validación de resultados experimentales
    - ReproducibilityChecker: Verificación de reproducibilidad
    - StatisticalValidator: Validación estadística de mejoras
    - BaselineComparator: Comparación con baselines SOTA
"""

from .result_validator import ResultValidator
from .reproducibility_checker import ReproducibilityChecker
from .statistical_validator import StatisticalValidator
from .baseline_comparator import BaselineComparator

__all__ = [
    'ResultValidator',
    'ReproducibilityChecker', 
    'StatisticalValidator',
    'BaselineComparator'
]

__version__ = '1.0.0' 