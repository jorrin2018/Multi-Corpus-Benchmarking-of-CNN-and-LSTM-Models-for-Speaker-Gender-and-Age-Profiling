#!/usr/bin/env python3
"""
Script de predicción para modelos de Speaker Profiling.

Este script permite realizar predicciones con modelos entrenados
para clasificación de género y edad en archivos de audio individuales.

Ejemplos de uso:
    # Predicción de género en un archivo
    python scripts/predict.py --model-path results/voxceleb1_gender_mobilenetv2_seed42/final_model.pth --audio-file sample.wav --task gender
    
    # Predicción de edad en múltiples archivos
    python scripts/predict.py --model-path results/common_voice_age_resnet18_seed42/final_model.pth --audio-dir samples/ --task age
    
    # Predicción con salida detallada
    python scripts/predict.py --model-path results/timit_age_lstm_256_2_seed42/final_model.pth --audio-file sample.wav --task age --output-probs
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Agregar el directorio src al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing.audio_processing import AudioProcessor
from preprocessing.feature_extraction import FeatureExtractor
from models.cnn_models import CNNModelFactory
from models.lstm_models import LSTMModelFactory
from utils.config_utils import ConfigManager


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
        description="Realizar predicciones con modelos de Speaker Profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Argumentos de modelo
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Ruta al modelo entrenado (.pth)"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        choices=["gender", "age"],
        required=True,
        help="Tarea de profiling (gender o age)"
    )
    
    # Argumentos de audio
    audio_group = parser.add_mutually_exclusive_group(required=True)
    audio_group.add_argument(
        "--audio-file",
        type=str,
        help="Archivo de audio individual"
    )
    audio_group.add_argument(
        "--audio-dir",
        type=str,
        help="Directorio con archivos de audio"
    )
    
    # Argumentos de configuración
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Archivo de configuración personalizada"
    )
    
    parser.add_argument(
        "--feature-type",
        type=str,
        choices=["spectrogram", "mel_spectrogram", "mfcc"],
        default="mel_spectrogram",
        help="Tipo de características a extraer (default: mel_spectrogram)"
    )
    
    # Argumentos de salida
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Archivo para guardar predicciones (CSV/YAML)"
    )
    
    parser.add_argument(
        "--output-probs",
        action="store_true",
        help="Incluir probabilidades en la salida"
    )
    
    parser.add_argument(
        "--output-features",
        action="store_true",
        help="Guardar características extraídas"
    )
    
    # Argumentos de hardware
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device a utilizar (default: auto)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño de batch para predicción (default: 32)"
    )
    
    # Argumentos de logging
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
    Configura el device para predicción.
    
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
        # Fallback para nombres no estándar
        dataset = "unknown"
        task = "unknown"
        model_name = "unknown"
        seed = "unknown"
    
    return {
        "dataset": dataset,
        "task": task,
        "model_name": model_name,
        "seed": seed
    }


def load_model(model_path: str, model_info: Dict, num_classes: int, input_shape: tuple, task: str, device: torch.device) -> nn.Module:
    """
    Carga un modelo entrenado.
    
    Args:
        model_path: Ruta al modelo
        model_info: Información del modelo
        num_classes: Número de clases
        input_shape: Forma del input
        task: Tarea
        device: Device
        
    Returns:
        Modelo cargado
    """
    model_name = model_info["model_name"]
    
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
    
    logging.info(f"Modelo cargado: {model_name}")
    return model


def find_audio_files(audio_path: str) -> List[str]:
    """
    Encuentra archivos de audio en un directorio.
    
    Args:
        audio_path: Ruta del directorio
        
    Returns:
        Lista de archivos de audio
    """
    audio_extensions = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
    audio_files = []
    
    path = Path(audio_path)
    for ext in audio_extensions:
        audio_files.extend(path.glob(f"*{ext}"))
        audio_files.extend(path.glob(f"**/*{ext}"))
    
    return sorted([str(f) for f in audio_files])


def extract_features(audio_file: str, feature_type: str, dataset: str, 
                    audio_processor: AudioProcessor, feature_extractor: FeatureExtractor) -> np.ndarray:
    """
    Extrae características de un archivo de audio.
    
    Args:
        audio_file: Archivo de audio
        feature_type: Tipo de características
        dataset: Dataset original del modelo
        audio_processor: Procesador de audio
        feature_extractor: Extractor de características
        
    Returns:
        Características extraídas
    """
    # Cargar y procesar audio
    audio_data = audio_processor.load_audio(audio_file)
    audio_data = audio_processor.preprocess_audio(audio_data)
    
    # Extraer características según el tipo
    if feature_type == "spectrogram":
        features = feature_extractor.extract_spectrogram(audio_data)
    elif feature_type == "mel_spectrogram":
        features = feature_extractor.extract_mel_spectrogram(audio_data, dataset=dataset)
    elif feature_type == "mfcc":
        features = feature_extractor.extract_mfcc(audio_data, dataset=dataset)
    else:
        raise ValueError(f"Tipo de características no soportado: {feature_type}")
    
    return features


def predict_single(model: nn.Module, features: np.ndarray, device: torch.device, 
                  task: str, output_probs: bool = False) -> Dict:
    """
    Realiza predicción en una muestra.
    
    Args:
        model: Modelo entrenado
        features: Características extraídas
        device: Device
        task: Tarea
        output_probs: Si incluir probabilidades
        
    Returns:
        Diccionario con predicción
    """
    # Convertir a tensor
    if len(features.shape) == 2:
        features = features[np.newaxis, ...]  # Agregar batch dimension
    
    features_tensor = torch.from_numpy(features).float().to(device)
    
    # Realizar predicción
    with torch.no_grad():
        outputs = model(features_tensor)
        
        if task == "age" and outputs.size(1) == 1:
            # Regresión de edad
            prediction = outputs.squeeze().cpu().numpy()
            if isinstance(prediction, np.ndarray) and prediction.shape == ():
                prediction = prediction.item()
            
            result = {
                "prediction": prediction,
                "prediction_type": "regression"
            }
            
            if output_probs:
                result["raw_output"] = outputs.squeeze().cpu().numpy().tolist()
        
        else:
            # Clasificación
            probs = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1)
            confidence = probs.max(dim=1)[0]
            
            result = {
                "prediction": predicted_class.cpu().numpy()[0],
                "confidence": confidence.cpu().numpy()[0],
                "prediction_type": "classification"
            }
            
            if output_probs:
                result["probabilities"] = probs.cpu().numpy()[0].tolist()
                result["raw_output"] = outputs.cpu().numpy()[0].tolist()
    
    return result


def predict_batch(model: nn.Module, features_list: List[np.ndarray], device: torch.device,
                 task: str, batch_size: int = 32, output_probs: bool = False) -> List[Dict]:
    """
    Realiza predicción en lotes.
    
    Args:
        model: Modelo entrenado
        features_list: Lista de características
        device: Device
        task: Tarea
        batch_size: Tamaño de batch
        output_probs: Si incluir probabilidades
        
    Returns:
        Lista de predicciones
    """
    results = []
    
    for i in tqdm(range(0, len(features_list), batch_size), desc="Prediciendo"):
        batch_features = features_list[i:i+batch_size]
        
        # Convertir a tensor
        batch_tensor = torch.from_numpy(np.array(batch_features)).float().to(device)
        
        # Realizar predicción
        with torch.no_grad():
            outputs = model(batch_tensor)
            
            if task == "age" and outputs.size(1) == 1:
                # Regresión de edad
                predictions = outputs.squeeze().cpu().numpy()
                if len(predictions.shape) == 0:
                    predictions = [predictions.item()]
                
                for j, pred in enumerate(predictions):
                    result = {
                        "prediction": pred,
                        "prediction_type": "regression"
                    }
                    
                    if output_probs:
                        result["raw_output"] = outputs[j].cpu().numpy().tolist()
                    
                    results.append(result)
            
            else:
                # Clasificación
                probs = F.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(outputs, dim=1)
                confidences = probs.max(dim=1)[0]
                
                for j in range(len(predicted_classes)):
                    result = {
                        "prediction": predicted_classes[j].cpu().numpy().item(),
                        "confidence": confidences[j].cpu().numpy().item(),
                        "prediction_type": "classification"
                    }
                    
                    if output_probs:
                        result["probabilities"] = probs[j].cpu().numpy().tolist()
                        result["raw_output"] = outputs[j].cpu().numpy().tolist()
                    
                    results.append(result)
    
    return results


def format_prediction_output(prediction: Dict, task: str, file_name: str = None) -> str:
    """
    Formatea la salida de predicción.
    
    Args:
        prediction: Predicción
        task: Tarea
        file_name: Nombre del archivo (opcional)
        
    Returns:
        Cadena formateada
    """
    output = []
    
    if file_name:
        output.append(f"Archivo: {file_name}")
    
    if task == "gender":
        gender_labels = ["Masculino", "Femenino"]
        if prediction["prediction_type"] == "classification":
            pred_label = gender_labels[prediction["prediction"]]
            confidence = prediction["confidence"]
            output.append(f"Género: {pred_label} (Confianza: {confidence:.3f})")
        else:
            output.append(f"Género: {prediction['prediction']}")
    
    elif task == "age":
        if prediction["prediction_type"] == "regression":
            age = prediction["prediction"]
            output.append(f"Edad: {age:.1f} años")
        else:
            # Clasificación por rangos de edad
            age_ranges = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
            if prediction["prediction"] < len(age_ranges):
                pred_label = age_ranges[prediction["prediction"]]
                confidence = prediction["confidence"]
                output.append(f"Rango de edad: {pred_label} (Confianza: {confidence:.3f})")
            else:
                output.append(f"Edad: {prediction['prediction']}")
    
    return " | ".join(output)


def save_results(results: List[Dict], output_file: str, audio_files: List[str]) -> None:
    """
    Guarda los resultados en un archivo.
    
    Args:
        results: Lista de resultados
        output_file: Archivo de salida
        audio_files: Lista de archivos de audio
    """
    output_path = Path(output_file)
    
    # Preparar datos
    data = []
    for i, (result, audio_file) in enumerate(zip(results, audio_files)):
        row = {
            "file": Path(audio_file).name,
            "file_path": audio_file,
            "prediction": result["prediction"],
            "prediction_type": result["prediction_type"]
        }
        
        if "confidence" in result:
            row["confidence"] = result["confidence"]
        
        if "probabilities" in result:
            for j, prob in enumerate(result["probabilities"]):
                row[f"prob_class_{j}"] = prob
        
        if "raw_output" in result:
            if isinstance(result["raw_output"], list):
                for j, val in enumerate(result["raw_output"]):
                    row[f"raw_output_{j}"] = val
            else:
                row["raw_output"] = result["raw_output"]
        
        data.append(row)
    
    # Guardar según extensión
    if output_path.suffix.lower() == '.csv':
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
    elif output_path.suffix.lower() in ['.yaml', '.yml']:
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    else:
        # Guardar como YAML por defecto
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    logging.info(f"Resultados guardados en: {output_path}")


def main():
    """Función principal del script de predicción."""
    args = parse_arguments()
    
    # Configurar logging
    setup_logging(args.log_level)
    
    logging.info("=" * 80)
    logging.info("INICIANDO PREDICCIÓN DE SPEAKER PROFILING")
    logging.info("=" * 80)
    
    # Configurar device
    device = setup_device(args.device)
    
    # Extraer información del modelo
    model_info = extract_model_info(args.model_path)
    logging.info(f"Modelo: {model_info['model_name']}")
    logging.info(f"Dataset original: {model_info['dataset']}")
    logging.info(f"Tarea: {args.task}")
    
    # Obtener archivos de audio
    if args.audio_file:
        audio_files = [args.audio_file]
    else:
        audio_files = find_audio_files(args.audio_dir)
    
    if not audio_files:
        logging.error("No se encontraron archivos de audio")
        sys.exit(1)
    
    logging.info(f"Procesando {len(audio_files)} archivos de audio")
    
    # Configurar procesadores
    config_manager = ConfigManager()
    if args.config:
        config = config_manager.load_config(args.config)
    else:
        config = config_manager.get_default_config()
    
    audio_processor = AudioProcessor(config["preprocessing"]["audio"])
    feature_extractor = FeatureExtractor(config["preprocessing"]["features"])
    
    # Extraer características
    logging.info("Extrayendo características...")
    features_list = []
    valid_files = []
    
    for audio_file in tqdm(audio_files, desc="Procesando archivos"):
        try:
            features = extract_features(
                audio_file, args.feature_type, model_info["dataset"],
                audio_processor, feature_extractor
            )
            features_list.append(features)
            valid_files.append(audio_file)
        except Exception as e:
            logging.warning(f"Error procesando {audio_file}: {e}")
            continue
    
    if not features_list:
        logging.error("No se pudieron procesar archivos de audio")
        sys.exit(1)
    
    # Determinar número de clases y forma del input
    sample_features = features_list[0]
    input_shape = sample_features.shape
    
    if args.task == "gender":
        num_classes = 2
    elif args.task == "age":
        if model_info["dataset"] == "timit":
            num_classes = 1  # Regresión
        else:
            num_classes = 6  # Clasificación por rangos
    
    # Cargar modelo
    try:
        model = load_model(
            args.model_path, model_info, num_classes, input_shape, args.task, device
        )
    except Exception as e:
        logging.error(f"Error cargando modelo: {e}")
        sys.exit(1)
    
    # Realizar predicciones
    logging.info("Realizando predicciones...")
    
    if len(features_list) == 1:
        # Predicción individual
        results = [predict_single(
            model, features_list[0], device, args.task, args.output_probs
        )]
    else:
        # Predicción en lotes
        results = predict_batch(
            model, features_list, device, args.task, args.batch_size, args.output_probs
        )
    
    # Mostrar resultados
    logging.info("=" * 80)
    logging.info("RESULTADOS DE PREDICCIÓN")
    logging.info("=" * 80)
    
    for i, (result, audio_file) in enumerate(zip(results, valid_files)):
        output = format_prediction_output(result, args.task, Path(audio_file).name)
        logging.info(output)
    
    # Guardar resultados si se especifica
    if args.output_file:
        save_results(results, args.output_file, valid_files)
    
    # Guardar características si se especifica
    if args.output_features:
        features_dir = Path("extracted_features")
        features_dir.mkdir(exist_ok=True)
        
        for i, (features, audio_file) in enumerate(zip(features_list, valid_files)):
            feature_file = features_dir / f"{Path(audio_file).stem}_features.npy"
            np.save(feature_file, features)
        
        logging.info(f"Características guardadas en: {features_dir}")
    
    logging.info("=" * 80)
    logging.info("PREDICCIÓN COMPLETADA")
    logging.info("=" * 80)


if __name__ == "__main__":
    main() 