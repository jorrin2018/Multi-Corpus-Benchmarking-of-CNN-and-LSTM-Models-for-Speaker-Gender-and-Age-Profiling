#!/usr/bin/env python3
"""
Script de evaluación para modelos de Speaker Profiling.

Este script permite evaluar modelos entrenados en diferentes datasets
y tareas, calculando métricas completas y realizando análisis estadístico.

Ejemplos de uso:
    # Evaluar modelo específico
    python scripts/evaluate.py --model-path results/voxceleb1_gender_mobilenetv2_seed42/final_model.pth --dataset voxceleb1 --task gender
    
    # Evaluar todos los modelos en un directorio
    python scripts/evaluate.py --models-dir results/ --dataset common_voice --task age
    
    # Evaluar con cross-corpus testing
    python scripts/evaluate.py --model-path results/voxceleb1_gender_mobilenetv2_seed42/final_model.pth --dataset common_voice --task gender --cross-corpus
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Agregar el directorio src al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets.voxceleb1 import VoxCeleb1Dataset
from datasets.common_voice import CommonVoiceDataset
from datasets.timit import TIMITDataset
from models.cnn_models import CNNModelFactory
from models.lstm_models import LSTMModelFactory
from evaluation.metrics import MetricsCalculator
from utils.config_utils import ConfigManager
from utils.data_utils import setup_data_loaders


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configura el sistema de logging.
    
    Args:
        log_level: Nivel de logging
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parsea los argumentos de línea de comandos.
    
    Returns:
        Argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description="Evaluar modelos de Speaker Profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Argumentos de modelo
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model-path",
        type=str,
        help="Ruta al modelo entrenado (.pth)"
    )
    model_group.add_argument(
        "--models-dir",
        type=str,
        help="Directorio con múltiples modelos entrenados"
    )
    
    # Argumentos de dataset
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["voxceleb1", "common_voice", "timit"],
        required=True,
        help="Dataset para evaluación"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        choices=["gender", "age"],
        required=True,
        help="Tarea de profiling (gender o age)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directorio de datos (default: data)"
    )
    
    # Argumentos de evaluación
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "val", "train"],
        default="test",
        help="Split a evaluar (default: test)"
    )
    
    parser.add_argument(
        "--cross-corpus",
        action="store_true",
        help="Realizar evaluación cross-corpus"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Tamaño de batch para evaluación (default: 64)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Número de workers para DataLoader (default: 4)"
    )
    
    # Argumentos de salida
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directorio de salida (default: evaluation_results)"
    )
    
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Guardar predicciones detalladas"
    )
    
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Guardar gráficos de evaluación"
    )
    
    # Argumentos de configuración
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device a utilizar (default: auto)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Nivel de logging (default: INFO)"
    )
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """
    Configura el device para evaluación.
    
    Args:
        device_arg: Argumento de device
        
    Returns:
        Device de PyTorch
    """
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    logging.info(f"Usando device: {device}")
    return device


def load_model(model_path: str, model_name: str, num_classes: int, input_shape: tuple, task: str, device: torch.device) -> nn.Module:
    """
    Carga un modelo entrenado.
    
    Args:
        model_path: Ruta al modelo
        model_name: Nombre del modelo
        num_classes: Número de clases
        input_shape: Forma del input
        task: Tarea
        device: Device
        
    Returns:
        Modelo cargado
    """
    # Crear modelo
    if model_name.startswith("lstm"):
        model = LSTMModelFactory.create_model(
            model_name=model_name,
            num_classes=num_classes,
            input_size=input_shape[-1],
            task=task
        )
    else:
        model = CNNModelFactory.create_model(
            model_name=model_name,
            num_classes=num_classes,
            input_shape=input_shape,
            task=task
        )
    
    # Cargar pesos
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def load_dataset(dataset_name: str, data_dir: str, task: str, split: str) -> tuple:
    """
    Carga el dataset especificado.
    
    Args:
        dataset_name: Nombre del dataset
        data_dir: Directorio de datos
        task: Tarea
        split: Split a cargar
        
    Returns:
        Dataset y data loader
    """
    data_path = Path(data_dir) / dataset_name
    
    if dataset_name == "voxceleb1":
        dataset = VoxCeleb1Dataset(
            data_dir=str(data_path),
            split=split,
            task=task,
            cache_features=True
        )
    elif dataset_name == "common_voice":
        dataset = CommonVoiceDataset(
            data_dir=str(data_path),
            split=split,
            task=task,
            cache_features=True
        )
    elif dataset_name == "timit":
        dataset = TIMITDataset(
            data_dir=str(data_path),
            split=split,
            task=task,
            cache_features=True
        )
    else:
        raise ValueError(f"Dataset no soportado: {dataset_name}")
    
    return dataset


def evaluate_model(model: nn.Module, dataset: torch.utils.data.Dataset, 
                  metrics_calculator: MetricsCalculator, device: torch.device,
                  batch_size: int = 64, num_workers: int = 4) -> Dict:
    """
    Evalúa un modelo en un dataset.
    
    Args:
        model: Modelo a evaluar
        dataset: Dataset de evaluación
        metrics_calculator: Calculadora de métricas
        device: Device
        batch_size: Tamaño de batch
        num_workers: Número de workers
        
    Returns:
        Diccionario con métricas
    """
    # Crear data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluando"):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            # Obtener predicciones
            if metrics_calculator.task == "age" and outputs.size(1) == 1:
                # Regresión
                predictions = outputs.squeeze()
            else:
                # Clasificación
                predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calcular métricas
    metrics = metrics_calculator.calculate_metrics(all_targets, all_predictions)
    
    return metrics, all_predictions, all_targets


def find_model_files(models_dir: str, dataset: str, task: str) -> List[str]:
    """
    Encuentra archivos de modelos en un directorio.
    
    Args:
        models_dir: Directorio de modelos
        dataset: Dataset
        task: Tarea
        
    Returns:
        Lista de rutas de modelos
    """
    models_path = Path(models_dir)
    pattern = f"{dataset}_{task}_*"
    
    model_files = []
    for exp_dir in models_path.glob(pattern):
        if exp_dir.is_dir():
            # Buscar archivos de modelo
            for model_file in ["final_model.pth", "best_model.pth", "model.pth"]:
                model_path = exp_dir / model_file
                if model_path.exists():
                    model_files.append(str(model_path))
                    break
    
    return sorted(model_files)


def extract_model_info(model_path: str) -> Dict:
    """
    Extrae información del modelo desde la ruta.
    
    Args:
        model_path: Ruta del modelo
        
    Returns:
        Diccionario con información del modelo
    """
    path = Path(model_path)
    exp_name = path.parent.name
    
    # Parsear nombre del experimento
    parts = exp_name.split("_")
    if len(parts) >= 4:
        dataset = parts[0]
        task = parts[1]
        model_name = "_".join(parts[2:-1])
        seed = parts[-1].replace("seed", "")
    else:
        dataset = "unknown"
        task = "unknown"
        model_name = "unknown"
        seed = "unknown"
    
    return {
        "dataset": dataset,
        "task": task,
        "model_name": model_name,
        "seed": seed,
        "exp_name": exp_name,
        "model_path": model_path
    }


def save_evaluation_results(results: List[Dict], output_dir: str, cross_corpus: bool = False) -> None:
    """
    Guarda los resultados de evaluación.
    
    Args:
        results: Lista de resultados
        output_dir: Directorio de salida
        cross_corpus: Si es evaluación cross-corpus
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Crear DataFrame
    df = pd.DataFrame(results)
    
    # Guardar CSV
    csv_name = "cross_corpus_results.csv" if cross_corpus else "evaluation_results.csv"
    df.to_csv(output_path / csv_name, index=False)
    
    # Guardar YAML
    yaml_name = "cross_corpus_results.yaml" if cross_corpus else "evaluation_results.yaml"
    with open(output_path / yaml_name, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    logging.info(f"Resultados guardados en: {output_path}")


def create_evaluation_plots(results: List[Dict], output_dir: str, task: str) -> None:
    """
    Crea gráficos de evaluación.
    
    Args:
        results: Resultados de evaluación
        output_dir: Directorio de salida
        task: Tarea
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Gráfico de barras por modelo
    plt.figure(figsize=(12, 6))
    
    if task == "gender":
        metric = "accuracy"
    elif task == "age":
        metric = "mae" if "mae" in df.columns else "accuracy"
    else:
        metric = "accuracy"
    
    if metric in df.columns:
        sns.barplot(data=df, x="model_name", y=metric)
        plt.title(f"Rendimiento por Modelo - {task.capitalize()}")
        plt.xlabel("Modelo")
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / f"performance_by_model_{task}.png", dpi=300)
        plt.close()
    
    # Gráfico de comparación de métricas
    if len(df) > 1:
        plt.figure(figsize=(10, 8))
        
        # Seleccionar métricas relevantes
        metric_cols = [col for col in df.columns if col not in ["model_name", "dataset", "task", "seed", "exp_name", "model_path"]]
        
        if metric_cols:
            # Normalizar datos para comparación
            df_norm = df[metric_cols].copy()
            for col in metric_cols:
                if col in ["accuracy", "f1_score", "precision", "recall"]:
                    # Métricas donde mayor es mejor
                    df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
                elif col in ["mae", "mse", "rmse"]:
                    # Métricas donde menor es mejor
                    df_norm[col] = (df_norm[col].max() - df_norm[col]) / (df_norm[col].max() - df_norm[col].min())
            
            # Crear heatmap
            df_norm["model_name"] = df["model_name"]
            df_pivot = df_norm.set_index("model_name")[metric_cols]
            
            sns.heatmap(df_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r')
            plt.title(f"Comparación de Métricas - {task.capitalize()}")
            plt.tight_layout()
            plt.savefig(output_path / f"metrics_comparison_{task}.png", dpi=300)
            plt.close()
    
    logging.info(f"Gráficos guardados en: {output_path}")


def main():
    """Función principal del script de evaluación."""
    args = parse_arguments()
    
    # Configurar logging
    setup_logging(args.log_level)
    
    logging.info("=" * 80)
    logging.info("INICIANDO EVALUACIÓN DE MODELOS")
    logging.info("=" * 80)
    
    # Configurar device
    device = setup_device(args.device)
    
    # Cargar dataset
    dataset = load_dataset(args.dataset, args.data_dir, args.task, args.split)
    
    # Obtener información del input
    sample_input, _ = dataset[0]
    input_shape = sample_input.shape
    
    # Determinar número de clases
    if args.task == "gender":
        num_classes = 2
    elif args.task == "age":
        if args.dataset == "timit":
            num_classes = 1  # Regresión
        else:
            num_classes = len(dataset.age_groups)  # Clasificación
    
    # Configurar calculadora de métricas
    metrics_calculator = MetricsCalculator(task=args.task)
    
    # Encontrar modelos a evaluar
    if args.model_path:
        model_files = [args.model_path]
    else:
        model_files = find_model_files(args.models_dir, args.dataset, args.task)
    
    if not model_files:
        logging.error("No se encontraron modelos para evaluar")
        sys.exit(1)
    
    logging.info(f"Evaluando {len(model_files)} modelos")
    
    # Evaluar modelos
    results = []
    
    for model_path in tqdm(model_files, desc="Evaluando modelos"):
        try:
            # Extraer información del modelo
            model_info = extract_model_info(model_path)
            model_name = model_info["model_name"]
            
            logging.info(f"Evaluando modelo: {model_name}")
            
            # Cargar modelo
            model = load_model(
                model_path, model_name, num_classes, input_shape, args.task, device
            )
            
            # Evaluar
            metrics, predictions, targets = evaluate_model(
                model, dataset, metrics_calculator, device, args.batch_size, args.num_workers
            )
            
            # Combinar resultados
            result = {**model_info, **metrics}
            results.append(result)
            
            # Guardar predicciones si se solicita
            if args.save_predictions:
                pred_dir = Path(args.output_dir) / "predictions"
                pred_dir.mkdir(parents=True, exist_ok=True)
                
                pred_file = pred_dir / f"{model_info['exp_name']}_predictions.csv"
                pred_df = pd.DataFrame({
                    "target": targets,
                    "prediction": predictions
                })
                pred_df.to_csv(pred_file, index=False)
            
            logging.info(f"Modelo evaluado exitosamente: {model_name}")
            
        except Exception as e:
            logging.error(f"Error evaluando modelo {model_path}: {e}")
            continue
    
    if not results:
        logging.error("No se pudieron evaluar modelos")
        sys.exit(1)
    
    # Guardar resultados
    save_evaluation_results(results, args.output_dir, args.cross_corpus)
    
    # Crear gráficos si se solicita
    if args.save_plots:
        create_evaluation_plots(results, args.output_dir, args.task)
    
    # Mostrar resumen
    logging.info("=" * 80)
    logging.info("RESUMEN DE EVALUACIÓN")
    logging.info("=" * 80)
    
    df = pd.DataFrame(results)
    
    # Mostrar mejores resultados
    if args.task == "gender":
        best_metric = "accuracy"
        best_model = df.loc[df[best_metric].idxmax()]
    elif args.task == "age":
        best_metric = "mae" if "mae" in df.columns else "accuracy"
        if best_metric == "mae":
            best_model = df.loc[df[best_metric].idxmin()]
        else:
            best_model = df.loc[df[best_metric].idxmax()]
    
    logging.info(f"Mejor modelo: {best_model['model_name']}")
    logging.info(f"Mejor {best_metric}: {best_model[best_metric]:.4f}")
    
    # Mostrar estadísticas
    logging.info("\nEstadísticas por modelo:")
    for model_name in df["model_name"].unique():
        model_results = df[df["model_name"] == model_name]
        if len(model_results) > 1:
            logging.info(f"{model_name}: {best_metric} = {model_results[best_metric].mean():.4f} ± {model_results[best_metric].std():.4f}")
        else:
            logging.info(f"{model_name}: {best_metric} = {model_results[best_metric].iloc[0]:.4f}")
    
    logging.info("=" * 80)
    logging.info("EVALUACIÓN COMPLETADA")
    logging.info("=" * 80)


if __name__ == "__main__":
    main() 