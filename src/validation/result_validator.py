"""
Validador principal de resultados experimentales.

Este módulo proporciona la clase ResultValidator que valida la integridad,
consistencia y validez científica de los resultados experimentales.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Resultado de una validación."""
    is_valid: bool
    score: float
    message: str
    details: Dict[str, Any]
    timestamp: str


@dataclass
class ExperimentResult:
    """Estructura de un resultado experimental."""
    model_name: str
    dataset: str
    task: str
    accuracy: float
    loss: float
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    mae: Optional[float] = None
    mse: Optional[float] = None
    seed: int = 42
    metadata: Optional[Dict[str, Any]] = None


class ResultValidator:
    """
    Validador principal de resultados experimentales.
    
    Esta clase proporciona métodos para validar la integridad, consistencia
    y validez científica de los resultados de experimentos de speaker profiling.
    """
    
    def __init__(self, 
                 baseline_file: Optional[str] = None,
                 tolerance: float = 0.05,
                 min_improvement: float = 0.001):
        """
        Inicializar el validador de resultados.
        
        Args:
            baseline_file: Archivo con resultados baseline para comparación
            tolerance: Tolerancia para variaciones de resultados (%)
            min_improvement: Mejora mínima considerada significativa (%)
        """
        self.baseline_file = baseline_file
        self.tolerance = tolerance
        self.min_improvement = min_improvement
        self.baselines = self._load_baselines()
        
        # Rangos válidos por tarea
        self.valid_ranges = {
            'gender': {'accuracy': (50.0, 100.0), 'loss': (0.0, 10.0)},
            'age': {'accuracy': (15.0, 80.0), 'loss': (0.0, 50.0)},
            'age_regression': {'mae': (1.0, 30.0), 'mse': (1.0, 900.0)}
        }
    
    def _load_baselines(self) -> Dict[str, Dict[str, float]]:
        """Cargar resultados baseline desde archivo."""
        if not self.baseline_file or not Path(self.baseline_file).exists():
            return self._get_default_baselines()
        
        try:
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error cargando baselines: {e}. Usando defaults.")
            return self._get_default_baselines()
    
    def _get_default_baselines(self) -> Dict[str, Dict[str, float]]:
        """Obtener baselines por defecto del paper."""
        return {
            'voxceleb1_gender': {
                'mobilenetv2': 97.8,
                'efficientnet_b0': 97.2,
                'resnet18': 96.9,
                'resnet50': 96.1,
                'vgg16': 95.4,
                'alexnet': 94.2,
                'densenet': 96.7,
                'lstm_128_1': 93.1,
                'lstm_128_2': 94.2,
                'lstm_128_3': 94.8,
                'lstm_256_1': 94.6,
                'lstm_256_2': 95.4,
                'lstm_256_3': 95.1,
                'lstm_512_1': 94.9,
                'lstm_512_2': 95.2,
                'lstm_512_3': 95.0
            },
            'common_voice_gender': {
                'mobilenetv2': 89.7,
                'efficientnet_b0': 88.3,
                'resnet18': 87.1,
                'resnet50': 86.8,
                'vgg16': 85.2,
                'alexnet': 83.9,
                'densenet': 87.4,
                'lstm_128_1': 81.4,
                'lstm_128_2': 82.1,
                'lstm_128_3': 82.9,
                'lstm_256_1': 83.2,
                'lstm_256_2': 84.2,
                'lstm_256_3': 83.8,
                'lstm_512_1': 83.6,
                'lstm_512_2': 83.9,
                'lstm_512_3': 84.2
            },
            'common_voice_age': {
                'mobilenetv2': 45.3,
                'efficientnet_b0': 43.8,
                'resnet18': 42.1,
                'resnet50': 41.9,
                'vgg16': 39.7,
                'alexnet': 37.2,
                'densenet': 42.8,
                'lstm_128_1': 36.1,
                'lstm_128_2': 37.2,
                'lstm_128_3': 38.1,
                'lstm_256_1': 37.8,
                'lstm_256_2': 38.9,
                'lstm_256_3': 38.4,
                'lstm_512_1': 38.2,
                'lstm_512_2': 38.6,
                'lstm_512_3': 38.9
            },
            'timit_gender': {
                'mobilenetv2': 98.2,
                'efficientnet_b0': 97.8,
                'resnet18': 97.4,
                'resnet50': 97.1,
                'vgg16': 96.8,
                'alexnet': 95.9,
                'densenet': 97.6,
                'lstm_128_1': 95.2,
                'lstm_128_2': 96.1,
                'lstm_128_3': 96.4,
                'lstm_256_1': 96.2,
                'lstm_256_2': 96.8,
                'lstm_256_3': 96.5,
                'lstm_512_1': 96.4,
                'lstm_512_2': 96.6,
                'lstm_512_3': 96.7
            }
        }
    
    def validate_single_result(self, result: ExperimentResult) -> ValidationResult:
        """
        Validar un resultado experimental individual.
        
        Args:
            result: Resultado experimental a validar
            
        Returns:
            ValidationResult con el resultado de la validación
        """
        validations = []
        
        # 1. Validar rangos de valores
        range_validation = self._validate_value_ranges(result)
        validations.append(range_validation)
        
        # 2. Validar consistencia de métricas
        consistency_validation = self._validate_metric_consistency(result)
        validations.append(consistency_validation)
        
        # 3. Comparar con baselines
        baseline_validation = self._validate_against_baseline(result)
        validations.append(baseline_validation)
        
        # 4. Validar metadatos
        metadata_validation = self._validate_metadata(result)
        validations.append(metadata_validation)
        
        # Calcular score total
        total_score = np.mean([v.score for v in validations])
        is_valid = all(v.is_valid for v in validations)
        
        return ValidationResult(
            is_valid=is_valid,
            score=total_score,
            message=self._generate_summary_message(validations),
            details={
                'range_validation': range_validation.__dict__,
                'consistency_validation': consistency_validation.__dict__,
                'baseline_validation': baseline_validation.__dict__,
                'metadata_validation': metadata_validation.__dict__
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _validate_value_ranges(self, result: ExperimentResult) -> ValidationResult:
        """Validar que los valores están en rangos esperados."""
        task_key = result.task
        if task_key not in self.valid_ranges:
            return ValidationResult(
                is_valid=False,
                score=0.0,
                message=f"Tarea desconocida: {task_key}",
                details={},
                timestamp=datetime.now().isoformat()
            )
        
        ranges = self.valid_ranges[task_key]
        issues = []
        
        # Validar accuracy si es tarea de clasificación
        if 'accuracy' in ranges and hasattr(result, 'accuracy'):
            min_acc, max_acc = ranges['accuracy']
            if not (min_acc <= result.accuracy <= max_acc):
                issues.append(f"Accuracy {result.accuracy}% fuera del rango [{min_acc}, {max_acc}]")
        
        # Validar loss
        if 'loss' in ranges:
            min_loss, max_loss = ranges['loss']
            if not (min_loss <= result.loss <= max_loss):
                issues.append(f"Loss {result.loss} fuera del rango [{min_loss}, {max_loss}]")
        
        # Validar MAE para regresión
        if 'mae' in ranges and result.mae is not None:
            min_mae, max_mae = ranges['mae']
            if not (min_mae <= result.mae <= max_mae):
                issues.append(f"MAE {result.mae} fuera del rango [{min_mae}, {max_mae}]")
        
        is_valid = len(issues) == 0
        score = 1.0 if is_valid else max(0.0, 1.0 - len(issues) * 0.3)
        
        return ValidationResult(
            is_valid=is_valid,
            score=score,
            message="Rangos válidos" if is_valid else "; ".join(issues),
            details={'issues': issues},
            timestamp=datetime.now().isoformat()
        )
    
    def _validate_metric_consistency(self, result: ExperimentResult) -> ValidationResult:
        """Validar consistencia entre métricas."""
        issues = []
        
        # Para clasificación, F1 debería estar relacionado con accuracy
        if (result.f1_score is not None and 
            hasattr(result, 'accuracy') and 
            result.task in ['gender', 'age']):
            
            # F1 y accuracy deberían estar correlacionados
            diff = abs(result.accuracy - result.f1_score * 100)
            if diff > 15:  # Tolerancia de 15%
                issues.append(f"F1 ({result.f1_score:.3f}) y accuracy ({result.accuracy:.1f}%) muy diferentes")
        
        # Precision y recall deberían ser razonables respecto a F1
        if (result.precision is not None and result.recall is not None and 
            result.f1_score is not None):
            
            expected_f1 = 2 * (result.precision * result.recall) / (result.precision + result.recall)
            f1_diff = abs(expected_f1 - result.f1_score)
            if f1_diff > 0.05:
                issues.append(f"F1 calculado ({expected_f1:.3f}) difiere del reportado ({result.f1_score:.3f})")
        
        # Para regresión, MAE y MSE deberían ser consistentes
        if result.mae is not None and result.mse is not None:
            # MSE debería ser >= MAE^2 (igualdad solo si todos los errores son iguales)
            if result.mse < (result.mae ** 2) * 0.9:
                issues.append(f"MSE ({result.mse:.3f}) inconsistente con MAE ({result.mae:.3f})")
        
        is_valid = len(issues) == 0
        score = 1.0 if is_valid else max(0.5, 1.0 - len(issues) * 0.2)
        
        return ValidationResult(
            is_valid=is_valid,
            score=score,
            message="Métricas consistentes" if is_valid else "; ".join(issues),
            details={'issues': issues},
            timestamp=datetime.now().isoformat()
        )
    
    def _validate_against_baseline(self, result: ExperimentResult) -> ValidationResult:
        """Validar resultado contra baselines conocidos."""
        task_dataset_key = f"{result.dataset}_{result.task}"
        
        if task_dataset_key not in self.baselines:
            return ValidationResult(
                is_valid=True,
                score=0.8,  # Score neutral para tareas sin baseline
                message=f"Sin baseline para {task_dataset_key}",
                details={},
                timestamp=datetime.now().isoformat()
            )
        
        baselines = self.baselines[task_dataset_key]
        
        if result.model_name not in baselines:
            return ValidationResult(
                is_valid=True,
                score=0.8,
                message=f"Sin baseline para modelo {result.model_name}",
                details={},
                timestamp=datetime.now().isoformat()
            )
        
        baseline_acc = baselines[result.model_name]
        actual_acc = result.accuracy
        
        # Calcular diferencia relativa
        diff = actual_acc - baseline_acc
        relative_diff = diff / baseline_acc
        
        # Determinar si es válido
        is_improvement = diff >= self.min_improvement
        is_within_tolerance = abs(relative_diff) <= self.tolerance
        is_valid = is_within_tolerance or is_improvement
        
        # Calcular score
        if is_improvement:
            score = min(1.0, 0.8 + (diff / baseline_acc) * 2)
        elif is_within_tolerance:
            score = 0.8
        else:
            score = max(0.2, 0.8 - abs(relative_diff) * 2)
        
        # Mensaje descriptivo
        if is_improvement:
            message = f"Mejora de {diff:.2f}% sobre baseline ({baseline_acc:.2f}%)"
        elif is_within_tolerance:
            message = f"Dentro de tolerancia del baseline ({baseline_acc:.2f}%)"
        else:
            message = f"Fuera de tolerancia: {actual_acc:.2f}% vs baseline {baseline_acc:.2f}%"
        
        return ValidationResult(
            is_valid=is_valid,
            score=score,
            message=message,
            details={
                'baseline_accuracy': baseline_acc,
                'actual_accuracy': actual_acc,
                'absolute_difference': diff,
                'relative_difference': relative_diff,
                'is_improvement': is_improvement
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _validate_metadata(self, result: ExperimentResult) -> ValidationResult:
        """Validar metadatos del experimento."""
        issues = []
        
        # Validar que se especificó el seed
        if not hasattr(result, 'seed') or result.seed is None:
            issues.append("Seed no especificado para reproducibilidad")
        
        # Validar metadatos adicionales
        if result.metadata:
            # Verificar que hay información de configuración
            if 'config' not in result.metadata:
                issues.append("Configuración del experimento no guardada")
            
            # Verificar información del dataset
            if 'dataset_size' not in result.metadata:
                issues.append("Tamaño del dataset no especificado")
            
            # Verificar información de entrenamiento
            if 'training_time' not in result.metadata:
                issues.append("Tiempo de entrenamiento no registrado")
        else:
            issues.append("Metadatos del experimento faltantes")
        
        is_valid = len(issues) == 0
        score = 1.0 if is_valid else max(0.6, 1.0 - len(issues) * 0.1)
        
        return ValidationResult(
            is_valid=is_valid,
            score=score,
            message="Metadatos completos" if is_valid else "; ".join(issues),
            details={'issues': issues},
            timestamp=datetime.now().isoformat()
        )
    
    def _generate_summary_message(self, validations: List[ValidationResult]) -> str:
        """Generar mensaje resumen de todas las validaciones."""
        valid_count = sum(1 for v in validations if v.is_valid)
        total_count = len(validations)
        
        if valid_count == total_count:
            return f"✅ Todas las validaciones pasaron ({valid_count}/{total_count})"
        else:
            return f"❌ {total_count - valid_count} validaciones fallaron ({valid_count}/{total_count})"
    
    def validate_experiment_batch(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """
        Validar un lote de resultados experimentales.
        
        Args:
            results: Lista de resultados experimentales
            
        Returns:
            Diccionario con resumen de validación del lote
        """
        validations = []
        for result in results:
            validation = self.validate_single_result(result)
            validations.append(validation)
        
        # Estadísticas del lote
        valid_count = sum(1 for v in validations if v.is_valid)
        total_count = len(validations)
        avg_score = np.mean([v.score for v in validations])
        
        # Agrupar por tipo de problema
        issues_by_type = {}
        for validation in validations:
            if not validation.is_valid:
                for detail_type, detail in validation.details.items():
                    if 'issues' in detail and detail['issues']:
                        if detail_type not in issues_by_type:
                            issues_by_type[detail_type] = []
                        issues_by_type[detail_type].extend(detail['issues'])
        
        return {
            'summary': {
                'total_results': total_count,
                'valid_results': valid_count,
                'invalid_results': total_count - valid_count,
                'average_score': avg_score,
                'success_rate': valid_count / total_count if total_count > 0 else 0
            },
            'validations': [v.__dict__ for v in validations],
            'issues_by_type': issues_by_type,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_validation_report(self, 
                             validation_results: Dict[str, Any], 
                             output_file: str) -> None:
        """
        Guardar reporte de validación en archivo.
        
        Args:
            validation_results: Resultados de validación
            output_file: Archivo de salida
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(validation_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Reporte de validación guardado en: {output_file}")
        except Exception as e:
            logger.error(f"Error guardando reporte: {e}")
            raise


def create_experiment_result_from_dict(data: Dict[str, Any]) -> ExperimentResult:
    """
    Crear ExperimentResult desde diccionario.
    
    Args:
        data: Diccionario con datos del experimento
        
    Returns:
        ExperimentResult inicializado
    """
    return ExperimentResult(
        model_name=data['model_name'],
        dataset=data['dataset'],
        task=data['task'],
        accuracy=data['accuracy'],
        loss=data['loss'],
        f1_score=data.get('f1_score'),
        precision=data.get('precision'),
        recall=data.get('recall'),
        mae=data.get('mae'),
        mse=data.get('mse'),
        seed=data.get('seed', 42),
        metadata=data.get('metadata')
    ) 