#!/usr/bin/env python3
"""
Script de benchmarking completo para Speaker Profiling.

Este script ejecuta experimentos completos multi-corpus con todos los modelos
CNN y LSTM especificados en el paper, incluyendo evaluación cross-corpus
y análisis estadístico comparativo.

Ejemplos de uso:
    # Benchmark completo en VoxCeleb1
    python scripts/benchmark.py --dataset voxceleb1 --task gender --seeds 42,123,456
    
    # Benchmark con modelos específicos
    python scripts/benchmark.py --dataset common_voice --task age --models mobilenetv2,resnet18,lstm_256_2
    
    # Benchmark con cross-corpus evaluation
    python scripts/benchmark.py --dataset voxceleb1 --task gender --cross-corpus --target-datasets common_voice,timit
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Agregar el directorio src al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config_utils import ConfigManager
from evaluation.metrics import MetricsCalculator


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configura el sistema de logging.
    
    Args:
        log_level: Nivel de logging
        log_file: Archivo de log (opcional)
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parsea los argumentos de línea de comandos.
    
    Returns:
        Argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description="Ejecutar benchmark completo de Speaker Profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Argumentos principales
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["voxceleb1", "common_voice", "timit"],
        required=True,
        help="Dataset principal para benchmark"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        choices=["gender", "age"],
        required=True,
        help="Tarea de profiling"
    )
    
    # Configuración de experimentos
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Modelos a evaluar (separados por comas) o 'all' para todos"
    )
    
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456,789,999,1337,2021,2022,2023,2024",
        help="Seeds para experimentos (separados por comas)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directorio de datos"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directorio de salida"
    )
    
    # Cross-corpus evaluation
    parser.add_argument(
        "--cross-corpus",
        action="store_true",
        help="Realizar evaluación cross-corpus"
    )
    
    parser.add_argument(
        "--target-datasets",
        type=str,
        default="",
        help="Datasets objetivo para cross-corpus (separados por comas)"
    )
    
    # Configuración de entrenamiento
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Número de épocas (usa config por defecto si no se especifica)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Tamaño de batch"
    )
    
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["selection", "finetune", "full"],
        default="full",
        help="Pipeline de entrenamiento"
    )
    
    # Configuración de ejecución
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=1,
        help="Número de jobs en paralelo"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device a utilizar"
    )
    
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Usar mixed precision training"
    )
    
    # Configuración de salida
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Guardar modelos entrenados"
    )
    
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Guardar gráficos de análisis"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Nivel de logging"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reanudar benchmark desde experimentos existentes"
    )
    
    return parser.parse_args()


def get_model_list(models_arg: str, task: str) -> List[str]:
    """
    Obtiene la lista de modelos a evaluar.
    
    Args:
        models_arg: Argumento de modelos
        task: Tarea
        
    Returns:
        Lista de nombres de modelos
    """
    if models_arg == "all":
        # Todos los modelos del paper
        cnn_models = [
            "mobilenetv2", "efficientnet_b0", "resnet50", "resnet18",
            "vgg16", "alexnet", "densenet"
        ]
        
        lstm_models = [
            "lstm_128_1", "lstm_128_2", "lstm_128_3",
            "lstm_256_1", "lstm_256_2", "lstm_256_3",
            "lstm_512_1", "lstm_512_2", "lstm_512_3"
        ]
        
        return cnn_models + lstm_models
    else:
        return [model.strip() for model in models_arg.split(",")]


def run_single_experiment(args: Tuple) -> Dict:
    """
    Ejecuta un experimento individual.
    
    Args:
        args: Tupla con argumentos del experimento
        
    Returns:
        Diccionario con resultados del experimento
    """
    (dataset, task, model, seed, config_dict, output_dir, 
     data_dir, save_models, device, mixed_precision, pipeline) = args
    
    import subprocess
    import json
    
    # Crear directorio de experimento
    exp_name = f"{dataset}_{task}_{model}_seed{seed}"
    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Verificar si el experimento ya fue ejecutado
    results_file = exp_dir / "final_results.yaml"
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = yaml.safe_load(f)
            results.update({
                "dataset": dataset,
                "task": task,
                "model": model,
                "seed": seed,
                "status": "completed",
                "exp_name": exp_name
            })
            return results
        except Exception as e:
            logging.warning(f"Error leyendo resultados existentes para {exp_name}: {e}")
    
    # Preparar comando de entrenamiento
    cmd = [
        sys.executable, "scripts/train.py",
        "--dataset", dataset,
        "--task", task,
        "--model", model,
        "--seed", str(seed),
        "--output-dir", str(exp_dir.parent),
        "--data-dir", data_dir,
        "--pipeline", pipeline,
        "--device", device,
        "--log-level", "WARNING"  # Reducir verbosidad
    ]
    
    if save_models:
        cmd.append("--save-model")
    
    if mixed_precision:
        cmd.append("--mixed-precision")
    
    # Configurar archivos adicionales
    if config_dict.get("batch_size"):
        cmd.extend(["--batch-size", str(config_dict["batch_size"])])
    
    if config_dict.get("epochs"):
        cmd.extend(["--epochs", str(config_dict["epochs"])])
    
    if config_dict.get("lr"):
        cmd.extend(["--lr", str(config_dict["lr"])])
    
    # Ejecutar entrenamiento
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 horas máximo
        end_time = time.time()
        
        if result.returncode == 0:
            # Leer resultados
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = yaml.safe_load(f)
            else:
                results = {"error": "No se encontraron resultados"}
            
            results.update({
                "dataset": dataset,
                "task": task,
                "model": model,
                "seed": seed,
                "status": "completed",
                "exp_name": exp_name,
                "training_time": end_time - start_time
            })
            
            return results
        else:
            logging.error(f"Error en experimento {exp_name}: {result.stderr}")
            return {
                "dataset": dataset,
                "task": task,
                "model": model,
                "seed": seed,
                "status": "failed",
                "exp_name": exp_name,
                "error": result.stderr
            }
    
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout en experimento {exp_name}")
        return {
            "dataset": dataset,
            "task": task,
            "model": model,
            "seed": seed,
            "status": "timeout",
            "exp_name": exp_name
        }
    
    except Exception as e:
        logging.error(f"Error ejecutando experimento {exp_name}: {e}")
        return {
            "dataset": dataset,
            "task": task,
            "model": model,
            "seed": seed,
            "status": "error",
            "exp_name": exp_name,
            "error": str(e)
        }


def run_cross_corpus_evaluation(source_results: List[Dict], target_datasets: List[str],
                               output_dir: str, data_dir: str, task: str) -> List[Dict]:
    """
    Ejecuta evaluación cross-corpus.
    
    Args:
        source_results: Resultados del dataset fuente
        target_datasets: Datasets objetivo
        output_dir: Directorio de salida
        data_dir: Directorio de datos
        task: Tarea
        
    Returns:
        Lista de resultados cross-corpus
    """
    import subprocess
    
    cross_corpus_results = []
    
    for result in source_results:
        if result["status"] != "completed":
            continue
        
        # Encontrar modelo entrenado
        model_path = Path(output_dir) / result["exp_name"] / "final_model.pth"
        if not model_path.exists():
            model_path = Path(output_dir) / result["exp_name"] / "best_model.pth"
        
        if not model_path.exists():
            logging.warning(f"No se encontró modelo para {result['exp_name']}")
            continue
        
        # Evaluar en cada dataset objetivo
        for target_dataset in target_datasets:
            if target_dataset == result["dataset"]:
                continue  # Saltar auto-evaluación
            
            try:
                # Ejecutar evaluación
                cmd = [
                    sys.executable, "scripts/evaluate.py",
                    "--model-path", str(model_path),
                    "--dataset", target_dataset,
                    "--task", task,
                    "--data-dir", data_dir,
                    "--output-dir", f"{output_dir}/cross_corpus",
                    "--cross-corpus",
                    "--log-level", "WARNING"
                ]
                
                result_eval = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                
                if result_eval.returncode == 0:
                    # Leer resultados cross-corpus
                    cross_results_file = Path(f"{output_dir}/cross_corpus/cross_corpus_results.yaml")
                    if cross_results_file.exists():
                        with open(cross_results_file, 'r') as f:
                            cross_results = yaml.safe_load(f)
                        
                        # Encontrar el resultado correspondiente
                        for cross_result in cross_results:
                            if cross_result.get("exp_name") == result["exp_name"]:
                                cross_result.update({
                                    "source_dataset": result["dataset"],
                                    "target_dataset": target_dataset,
                                    "cross_corpus": True
                                })
                                cross_corpus_results.append(cross_result)
                                break
                
            except Exception as e:
                logging.error(f"Error en evaluación cross-corpus {result['exp_name']} -> {target_dataset}: {e}")
    
    return cross_corpus_results


def analyze_results(results: List[Dict], task: str) -> Dict:
    """
    Analiza los resultados del benchmark.
    
    Args:
        results: Lista de resultados
        task: Tarea
        
    Returns:
        Diccionario con análisis estadístico
    """
    df = pd.DataFrame(results)
    
    # Filtrar resultados exitosos
    successful_results = df[df["status"] == "completed"]
    
    if len(successful_results) == 0:
        return {"error": "No hay resultados exitosos para analizar"}
    
    # Determinar métrica principal
    if task == "gender":
        primary_metric = "accuracy"
    elif task == "age":
        primary_metric = "mae" if "mae" in successful_results.columns else "accuracy"
    else:
        primary_metric = "accuracy"
    
    analysis = {
        "total_experiments": len(results),
        "successful_experiments": len(successful_results),
        "failed_experiments": len(df[df["status"] == "failed"]),
        "primary_metric": primary_metric
    }
    
    if primary_metric in successful_results.columns:
        # Análisis por modelo
        model_stats = successful_results.groupby("model")[primary_metric].agg([
            "count", "mean", "std", "min", "max"
        ]).round(4)
        
        analysis["model_statistics"] = model_stats.to_dict()
        
        # Mejor modelo
        if primary_metric in ["accuracy", "f1_score", "precision", "recall"]:
            best_model = model_stats["mean"].idxmax()
            best_score = model_stats["mean"].max()
        else:
            best_model = model_stats["mean"].idxmin()
            best_score = model_stats["mean"].min()
        
        analysis["best_model"] = {
            "name": best_model,
            "score": best_score,
            "metric": primary_metric
        }
        
        # Análisis por dataset (si hay cross-corpus)
        if "source_dataset" in successful_results.columns:
            dataset_stats = successful_results.groupby(["source_dataset", "target_dataset"])[primary_metric].agg([
                "count", "mean", "std"
            ]).round(4)
            
            analysis["cross_corpus_statistics"] = dataset_stats.to_dict()
        
        # Tests estadísticos
        if len(successful_results["model"].unique()) > 1:
            # ANOVA para comparar modelos
            model_groups = [group[primary_metric].values for name, group in successful_results.groupby("model")]
            
            if len(model_groups) > 1 and all(len(group) > 1 for group in model_groups):
                try:
                    f_stat, p_value = stats.f_oneway(*model_groups)
                    analysis["anova_test"] = {
                        "f_statistic": f_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    }
                except:
                    analysis["anova_test"] = {"error": "No se pudo realizar ANOVA"}
    
    return analysis


def create_benchmark_plots(results: List[Dict], analysis: Dict, output_dir: str, task: str) -> None:
    """
    Crea gráficos de análisis del benchmark.
    
    Args:
        results: Lista de resultados
        analysis: Análisis estadístico
        output_dir: Directorio de salida
        task: Tarea
    """
    df = pd.DataFrame(results)
    successful_results = df[df["status"] == "completed"]
    
    if len(successful_results) == 0:
        logging.warning("No hay resultados exitosos para crear gráficos")
        return
    
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    primary_metric = analysis.get("primary_metric", "accuracy")
    
    # Gráfico 1: Rendimiento por modelo
    if primary_metric in successful_results.columns:
        plt.figure(figsize=(15, 8))
        
        # Box plot
        sns.boxplot(data=successful_results, x="model", y=primary_metric)
        plt.title(f"Distribución de {primary_metric.upper()} por Modelo - {task.capitalize()}")
        plt.xlabel("Modelo")
        plt.ylabel(primary_metric.upper())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / f"performance_by_model_{task}.png", dpi=300)
        plt.close()
        
        # Gráfico 2: Comparación de medias
        plt.figure(figsize=(12, 6))
        
        model_means = successful_results.groupby("model")[primary_metric].mean().sort_values(ascending=False)
        model_stds = successful_results.groupby("model")[primary_metric].std()
        
        plt.bar(range(len(model_means)), model_means.values, yerr=model_stds.values, capsize=5)
        plt.title(f"Rendimiento Promedio por Modelo - {task.capitalize()}")
        plt.xlabel("Modelo")
        plt.ylabel(f"{primary_metric.upper()} Promedio")
        plt.xticks(range(len(model_means)), model_means.index, rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / f"average_performance_{task}.png", dpi=300)
        plt.close()
    
    # Gráfico 3: Variabilidad por modelo
    if "model" in successful_results.columns and len(successful_results) > 10:
        plt.figure(figsize=(12, 6))
        
        model_vars = successful_results.groupby("model")[primary_metric].std().sort_values()
        
        plt.bar(range(len(model_vars)), model_vars.values)
        plt.title(f"Variabilidad (Desviación Estándar) por Modelo - {task.capitalize()}")
        plt.xlabel("Modelo")
        plt.ylabel(f"Desviación Estándar de {primary_metric.upper()}")
        plt.xticks(range(len(model_vars)), model_vars.index, rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / f"variability_by_model_{task}.png", dpi=300)
        plt.close()
    
    # Gráfico 4: Cross-corpus si está disponible
    if "source_dataset" in successful_results.columns and "target_dataset" in successful_results.columns:
        plt.figure(figsize=(10, 8))
        
        # Crear matriz de resultados cross-corpus
        cross_results = successful_results.groupby(["source_dataset", "target_dataset"])[primary_metric].mean().unstack()
        
        sns.heatmap(cross_results, annot=True, fmt='.3f', cmap='RdYlBu_r')
        plt.title(f"Evaluación Cross-Corpus - {task.capitalize()}")
        plt.xlabel("Dataset Objetivo")
        plt.ylabel("Dataset Fuente")
        plt.tight_layout()
        plt.savefig(plots_dir / f"cross_corpus_heatmap_{task}.png", dpi=300)
        plt.close()
    
    # Gráfico 5: Tiempo de entrenamiento si está disponible
    if "training_time" in successful_results.columns:
        plt.figure(figsize=(12, 6))
        
        training_times = successful_results.groupby("model")["training_time"].mean().sort_values()
        
        plt.bar(range(len(training_times)), training_times.values / 60)  # Convertir a minutos
        plt.title(f"Tiempo de Entrenamiento Promedio por Modelo - {task.capitalize()}")
        plt.xlabel("Modelo")
        plt.ylabel("Tiempo (minutos)")
        plt.xticks(range(len(training_times)), training_times.index, rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / f"training_time_{task}.png", dpi=300)
        plt.close()
    
    logging.info(f"Gráficos guardados en: {plots_dir}")


def main():
    """Función principal del script de benchmark."""
    args = parse_arguments()
    
    # Configurar logging
    log_file = Path(args.output_dir) / "benchmark.log" if args.output_dir else None
    setup_logging(args.log_level, log_file)
    
    logging.info("=" * 80)
    logging.info("INICIANDO BENCHMARK DE SPEAKER PROFILING")
    logging.info("=" * 80)
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Tarea: {args.task}")
    logging.info(f"Pipeline: {args.pipeline}")
    
    # Obtener configuración
    config_manager = ConfigManager()
    config = config_manager.get_default_config()
    
    # Sobrescribir configuración con argumentos
    config_dict = {}
    if args.batch_size:
        config_dict["batch_size"] = args.batch_size
    if args.epochs:
        config_dict["epochs"] = args.epochs
    
    # Obtener listas de modelos y seeds
    models = get_model_list(args.models, args.task)
    seeds = [int(seed.strip()) for seed in args.seeds.split(",")]
    
    logging.info(f"Modelos: {len(models)} ({', '.join(models)})")
    logging.info(f"Seeds: {len(seeds)} ({', '.join(map(str, seeds))})")
    logging.info(f"Total de experimentos: {len(models) * len(seeds)}")
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Preparar experimentos
    experiments = []
    for model in models:
        for seed in seeds:
            exp_args = (
                args.dataset, args.task, model, seed, config_dict,
                str(output_dir), args.data_dir, args.save_models,
                args.device, args.mixed_precision, args.pipeline
            )
            experiments.append(exp_args)
    
    # Ejecutar experimentos
    logging.info("Ejecutando experimentos...")
    results = []
    
    if args.parallel_jobs > 1:
        # Ejecución paralela
        with ProcessPoolExecutor(max_workers=args.parallel_jobs) as executor:
            futures = {executor.submit(run_single_experiment, exp): exp for exp in experiments}
            
            for future in tqdm(as_completed(futures), total=len(experiments), desc="Experimentos"):
                result = future.result()
                results.append(result)
                
                # Log progreso
                if result["status"] == "completed":
                    logging.info(f"✓ Completado: {result['exp_name']}")
                else:
                    logging.warning(f"✗ Falló: {result['exp_name']} - {result['status']}")
    else:
        # Ejecución secuencial
        for exp in tqdm(experiments, desc="Experimentos"):
            result = run_single_experiment(exp)
            results.append(result)
            
            if result["status"] == "completed":
                logging.info(f"✓ Completado: {result['exp_name']}")
            else:
                logging.warning(f"✗ Falló: {result['exp_name']} - {result['status']}")
    
    # Cross-corpus evaluation
    if args.cross_corpus and args.target_datasets:
        logging.info("Ejecutando evaluación cross-corpus...")
        target_datasets = [ds.strip() for ds in args.target_datasets.split(",")]
        
        cross_corpus_results = run_cross_corpus_evaluation(
            results, target_datasets, str(output_dir), args.data_dir, args.task
        )
        
        results.extend(cross_corpus_results)
        logging.info(f"Completadas {len(cross_corpus_results)} evaluaciones cross-corpus")
    
    # Análisis de resultados
    logging.info("Analizando resultados...")
    analysis = analyze_results(results, args.task)
    
    # Guardar resultados
    results_file = output_dir / "benchmark_results.yaml"
    with open(results_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    analysis_file = output_dir / "benchmark_analysis.yaml"
    with open(analysis_file, 'w') as f:
        yaml.dump(analysis, f, default_flow_style=False)
    
    # Crear CSV para análisis
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "benchmark_results.csv", index=False)
    
    # Crear gráficos
    if args.save_plots:
        create_benchmark_plots(results, analysis, str(output_dir), args.task)
    
    # Mostrar resumen
    logging.info("=" * 80)
    logging.info("RESUMEN DEL BENCHMARK")
    logging.info("=" * 80)
    
    total_experiments = len(results)
    successful = len([r for r in results if r["status"] == "completed"])
    failed = len([r for r in results if r["status"] != "completed"])
    
    logging.info(f"Total de experimentos: {total_experiments}")
    logging.info(f"Exitosos: {successful}")
    logging.info(f"Fallidos: {failed}")
    logging.info(f"Tasa de éxito: {successful/total_experiments*100:.1f}%")
    
    if "best_model" in analysis:
        best = analysis["best_model"]
        logging.info(f"Mejor modelo: {best['name']}")
        logging.info(f"Mejor {best['metric']}: {best['score']:.4f}")
    
    logging.info(f"Resultados guardados en: {output_dir}")
    logging.info("=" * 80)
    logging.info("BENCHMARK COMPLETADO")
    logging.info("=" * 80)


if __name__ == "__main__":
    main() 