#!/usr/bin/env python3
"""
Script principal de entrenamiento para modelos de Speaker Profiling.

Este script permite entrenar modelos CNN y LSTM en los datasets VoxCeleb1, 
Common Voice y TIMIT siguiendo el pipeline de 2 etapas del paper:
1. Etapa de selección de modelo
2. Etapa de fine-tuning

Ejemplos de uso:
    # Entrenar CNN en VoxCeleb1 para clasificación de género
    python scripts/train.py --dataset voxceleb1 --task gender --model mobilenetv2 --seed 42
    
    # Entrenar LSTM en Common Voice para clasificación de edad
    python scripts/train.py --dataset common_voice --task age --model lstm_256_2 --seed 42
    
    # Entrenar con pipeline completo de 2 etapas
    python scripts/train.py --dataset timit --task age --model resnet18 --pipeline full --seed 42
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# Agregar el directorio src al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets.voxceleb1 import VoxCeleb1Dataset
from datasets.common_voice import CommonVoiceDataset
from datasets.timit import TIMITDataset
from models.cnn_models import CNNModelFactory
from models.lstm_models import LSTMModelFactory
from training.trainer import SpeakerProfilingTrainer
from training.callbacks import CallbackManager
from utils.config_utils import ConfigManager
from utils.data_utils import setup_data_loaders
from evaluation.metrics import MetricsCalculator


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configura el sistema de logging.
    
    Args:
        log_level: Nivel de logging ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Archivo donde guardar los logs (opcional)
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parsea los argumentos de línea de comandos.
    
    Returns:
        Argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description="Entrenar modelos de Speaker Profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Argumentos principales
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["voxceleb1", "common_voice", "timit"],
        required=True,
        help="Dataset a utilizar"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        choices=["gender", "age"],
        required=True,
        help="Tarea de profiling (gender o age)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Modelo a utilizar (ej: mobilenetv2, lstm_256_2, resnet18)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad (default: 42)"
    )
    
    # Configuración de datos
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directorio de datos (default: data)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Tamaño de batch (usa config por defecto si no se especifica)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Número de workers para DataLoader (default: 4)"
    )
    
    # Configuración de entrenamiento
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Número de épocas (usa config por defecto si no se especifica)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (usa config por defecto si no se especifica)"
    )
    
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["selection", "finetune", "full"],
        default="full",
        help="Pipeline a ejecutar: selection, finetune, o full (default: full)"
    )
    
    # Configuración de salida
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directorio de salida (default: results)"
    )
    
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Guardar modelo entrenado"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Nivel de logging (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Archivo para guardar logs (opcional)"
    )
    
    # Configuración de hardware
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device a utilizar (default: auto)"
    )
    
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Usar mixed precision training"
    )
    
    # Configuración personalizada
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Archivo de configuración personalizada"
    )
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """
    Configura el device para entrenamiento.
    
    Args:
        device_arg: Argumento de device ('auto', 'cpu', 'cuda')
        
    Returns:
        Device de PyTorch
    """
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    logging.info(f"Usando device: {device}")
    
    if device.type == "cuda":
        logging.info(f"GPU disponible: {torch.cuda.get_device_name()}")
        logging.info(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def load_dataset(dataset_name: str, data_dir: str, task: str) -> tuple:
    """
    Carga el dataset especificado.
    
    Args:
        dataset_name: Nombre del dataset
        data_dir: Directorio de datos
        task: Tarea (gender/age)
        
    Returns:
        Tuple con (train_dataset, val_dataset, test_dataset)
    """
    data_path = Path(data_dir) / dataset_name
    
    if dataset_name == "voxceleb1":
        train_dataset = VoxCeleb1Dataset(
            data_dir=str(data_path),
            split="train",
            task=task,
            cache_features=True
        )
        val_dataset = VoxCeleb1Dataset(
            data_dir=str(data_path),
            split="val",
            task=task,
            cache_features=True
        )
        test_dataset = VoxCeleb1Dataset(
            data_dir=str(data_path),
            split="test",
            task=task,
            cache_features=True
        )
    
    elif dataset_name == "common_voice":
        train_dataset = CommonVoiceDataset(
            data_dir=str(data_path),
            split="train",
            task=task,
            cache_features=True
        )
        val_dataset = CommonVoiceDataset(
            data_dir=str(data_path),
            split="val",
            task=task,
            cache_features=True
        )
        test_dataset = CommonVoiceDataset(
            data_dir=str(data_path),
            split="test",
            task=task,
            cache_features=True
        )
    
    elif dataset_name == "timit":
        train_dataset = TIMITDataset(
            data_dir=str(data_path),
            split="train",
            task=task,
            cache_features=True
        )
        val_dataset = TIMITDataset(
            data_dir=str(data_path),
            split="val",
            task=task,
            cache_features=True
        )
        test_dataset = TIMITDataset(
            data_dir=str(data_path),
            split="test",
            task=task,
            cache_features=True
        )
    
    else:
        raise ValueError(f"Dataset no soportado: {dataset_name}")
    
    logging.info(f"Dataset cargado - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def create_model(model_name: str, num_classes: int, input_shape: tuple, task: str) -> nn.Module:
    """
    Crea el modelo especificado.
    
    Args:
        model_name: Nombre del modelo
        num_classes: Número de clases
        input_shape: Forma del input
        task: Tarea (gender/age)
        
    Returns:
        Modelo de PyTorch
    """
    # Determinar si es CNN o LSTM
    if model_name.startswith("lstm"):
        # Modelo LSTM
        model = LSTMModelFactory.create_model(
            model_name=model_name,
            num_classes=num_classes,
            input_size=input_shape[-1],  # Última dimensión para LSTM
            task=task
        )
    else:
        # Modelo CNN
        model = CNNModelFactory.create_model(
            model_name=model_name,
            num_classes=num_classes,
            input_shape=input_shape,
            task=task
        )
    
    logging.info(f"Modelo creado: {model_name}")
    logging.info(f"Número de parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def main():
    """Función principal del script de entrenamiento."""
    args = parse_arguments()
    
    # Configurar logging
    setup_logging(args.log_level, args.log_file)
    
    logging.info("=" * 80)
    logging.info("INICIANDO ENTRENAMIENTO DE SPEAKER PROFILING")
    logging.info("=" * 80)
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Tarea: {args.task}")
    logging.info(f"Modelo: {args.model}")
    logging.info(f"Seed: {args.seed}")
    logging.info(f"Pipeline: {args.pipeline}")
    
    # Configurar reproducibilidad
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Configurar device
    device = setup_device(args.device)
    
    # Cargar configuración
    config_manager = ConfigManager()
    
    if args.config:
        config = config_manager.load_config(args.config)
    else:
        config = config_manager.get_default_config()
    
    # Sobrescribir configuración con argumentos de línea de comandos
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    
    # Crear directorios de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_dir = output_dir / f"{args.dataset}_{args.task}_{args.model}_seed{args.seed}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar dataset
    try:
        train_dataset, val_dataset, test_dataset = load_dataset(
            args.dataset, args.data_dir, args.task
        )
    except Exception as e:
        logging.error(f"Error al cargar dataset: {e}")
        sys.exit(1)
    
    # Configurar data loaders
    train_loader, val_loader, test_loader = setup_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=args.num_workers,
        shuffle=True
    )
    
    # Determinar número de clases
    if args.task == "gender":
        num_classes = 2
    elif args.task == "age":
        if args.dataset == "timit":
            num_classes = 1  # Regresión
        else:
            num_classes = len(train_dataset.age_groups)  # Clasificación
    
    # Obtener forma del input
    sample_input, _ = train_dataset[0]
    input_shape = sample_input.shape
    
    # Crear modelo
    try:
        model = create_model(args.model, num_classes, input_shape, args.task)
        model = model.to(device)
    except Exception as e:
        logging.error(f"Error al crear modelo: {e}")
        sys.exit(1)
    
    # Configurar trainer
    trainer = SpeakerProfilingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        output_dir=str(experiment_dir),
        use_mixed_precision=args.mixed_precision
    )
    
    # Configurar callbacks
    callback_manager = CallbackManager(
        output_dir=str(experiment_dir),
        save_best_model=args.save_model,
        patience=config["training"]["early_stopping"]["patience"],
        monitor_metric="val_loss"
    )
    
    trainer.set_callbacks(callback_manager)
    
    # Entrenar según pipeline
    try:
        if args.pipeline == "selection":
            logging.info("Ejecutando etapa de selección de modelo...")
            trainer.train_selection_stage()
            
        elif args.pipeline == "finetune":
            logging.info("Ejecutando etapa de fine-tuning...")
            # Cargar modelo pre-entrenado si existe
            model_path = experiment_dir / "best_model_selection.pth"
            if model_path.exists():
                trainer.load_model(str(model_path))
            else:
                logging.warning("No se encontró modelo de selección, usando modelo inicial")
            
            trainer.train_finetune_stage()
            
        elif args.pipeline == "full":
            logging.info("Ejecutando pipeline completo de 2 etapas...")
            trainer.train_full_pipeline()
        
        # Evaluación final
        logging.info("Realizando evaluación final...")
        metrics_calculator = MetricsCalculator(task=args.task)
        
        test_metrics = trainer.evaluate(test_loader, metrics_calculator)
        
        logging.info("=" * 80)
        logging.info("RESULTADOS FINALES")
        logging.info("=" * 80)
        
        for metric_name, metric_value in test_metrics.items():
            logging.info(f"{metric_name}: {metric_value:.4f}")
        
        # Guardar resultados
        results_file = experiment_dir / "final_results.yaml"
        with open(results_file, 'w') as f:
            yaml.dump(test_metrics, f, default_flow_style=False)
        
        logging.info(f"Resultados guardados en: {results_file}")
        
        if args.save_model:
            model_file = experiment_dir / "final_model.pth"
            trainer.save_model(str(model_file))
            logging.info(f"Modelo guardado en: {model_file}")
        
    except Exception as e:
        logging.error(f"Error durante el entrenamiento: {e}")
        sys.exit(1)
    
    logging.info("=" * 80)
    logging.info("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    logging.info("=" * 80)


if __name__ == "__main__":
    main() 