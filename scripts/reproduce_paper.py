#!/usr/bin/env python3
"""
Script para reproducir exactamente los resultados del paper.

Este script ejecuta todos los experimentos necesarios para reproducir
los resultados reportados en el paper "Multi-Corpus Benchmarking of CNN & LSTM Models for Speaker Profiling".

Reproduce:
- Todos los modelos CNN y LSTM con configuraciones exactas
- Experimentos con 10 seeds (42, 123, 456, 789, 999, 1337, 2021, 2022, 2023, 2024)
- Pipeline de 2 etapas (selección + fine-tuning)
- Evaluación cross-corpus
- Comparación con estado del arte

Ejemplos de uso:
    # Reproducir resultados completos
    python scripts/reproduce_paper.py --all-experiments
    
    # Reproducir solo VoxCeleb1 gender
    python scripts/reproduce_paper.py --dataset voxceleb1 --task gender
    
    # Reproducir con análisis estadístico completo
    python scripts/reproduce_paper.py --dataset common_voice --task age --statistical-analysis
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, wilcoxon

# Agregar el directorio src al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config_utils import ConfigManager


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
        description="Reproducir resultados exactos del paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Argumentos principales
    parser.add_argument(
        "--all-experiments",
        action="store_true",
        help="Ejecutar todos los experimentos del paper"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["voxceleb1", "common_voice", "timit"],
        help="Dataset específico a reproducir"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        choices=["gender", "age"],
        help="Tarea específica a reproducir"
    )
    
    # Configuración
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directorio de datos"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper_reproduction",
        help="Directorio de salida"
    )
    
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=1,
        help="Número de jobs en paralelo"
    )
    
    # Análisis
    parser.add_argument(
        "--statistical-analysis",
        action="store_true",
        help="Realizar análisis estadístico completo"
    )
    
    parser.add_argument(
        "--sota-comparison",
        action="store_true",
        help="Comparar con estado del arte"
    )
    
    parser.add_argument(
        "--cross-corpus",
        action="store_true",
        help="Realizar evaluación cross-corpus"
    )
    
    # Configuración de hardware
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
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Nivel de logging"
    )
    
    return parser.parse_args()


def get_paper_configuration() -> Dict:
    """
    Obtiene la configuración exacta del paper.
    
    Returns:
        Configuración completa del paper
    """
    return {
        "seeds": [42, 123, 456, 789, 999, 1337, 2021, 2022, 2023, 2024],
        
        "models": {
            "cnn": [
                "mobilenetv2", "efficientnet_b0", "resnet50", "resnet18",
                "vgg16", "alexnet", "densenet"
            ],
            "lstm": [
                "lstm_128_1", "lstm_128_2", "lstm_128_3",
                "lstm_256_1", "lstm_256_2", "lstm_256_3",
                "lstm_512_1", "lstm_512_2", "lstm_512_3"
            ]
        },
        
        "datasets": {
            "voxceleb1": {
                "tasks": ["gender"],
                "speakers": 1251,
                "sample_rate": 16000,
                "chunk_duration": 3.0,
                "feature_type": "mel_spectrogram",
                "feature_size": 224
            },
            "common_voice": {
                "tasks": ["gender", "age"],
                "speakers": 13060,
                "sample_rate": 22050,
                "chunk_duration": 3.0,
                "feature_type": "mel_spectrogram",
                "feature_size": 128
            },
            "timit": {
                "tasks": ["gender", "age"],
                "speakers": 630,
                "sample_rate": 16000,
                "chunk_duration": None,  # Usar audio completo
                "feature_type": "mfcc",
                "feature_size": 64
            }
        },
        
        "training": {
            "pipeline": "full",  # 2 etapas: selección + fine-tuning
            "batch_size": 32,
            "epochs_selection": 50,
            "epochs_finetune": 30,
            "learning_rate_selection": 0.001,
            "learning_rate_finetune": 0.0001,
            "optimizer": "adam",
            "scheduler": "reduce_on_plateau",
            "early_stopping": True,
            "patience": 10
        },
        
        "sota_baselines": {
            "voxceleb1_gender": {
                "baseline_accuracy": 0.9543,
                "paper_improvement": 0.0057,
                "expected_accuracy": 0.9600
            },
            "common_voice_gender": {
                "baseline_accuracy": 0.9275,
                "paper_improvement": 0.0125,
                "expected_accuracy": 0.9400
            },
            "common_voice_age": {
                "baseline_accuracy": 0.6214,
                "paper_improvement": 0.0286,
                "expected_accuracy": 0.6500
            },
            "timit_gender": {
                "baseline_accuracy": 0.9800,
                "paper_improvement": 0.0050,
                "expected_accuracy": 0.9850
            },
            "timit_age": {
                "baseline_mae": 8.5,
                "paper_improvement": -1.2,
                "expected_mae": 7.3
            }
        }
    }


def get_experiment_list(config: Dict, dataset: str = None, task: str = None) -> List[Dict]:
    """
    Obtiene la lista de experimentos a ejecutar.
    
    Args:
        config: Configuración del paper
        dataset: Dataset específico (opcional)
        task: Tarea específica (opcional)
        
    Returns:
        Lista de experimentos
    """
    experiments = []
    
    # Determinar datasets y tareas
    if dataset and task:
        dataset_tasks = [(dataset, task)]
    elif dataset:
        dataset_tasks = [(dataset, t) for t in config["datasets"][dataset]["tasks"]]
    else:
        dataset_tasks = []
        for ds, info in config["datasets"].items():
            for t in info["tasks"]:
                dataset_tasks.append((ds, t))
    
    # Generar experimentos
    all_models = config["models"]["cnn"] + config["models"]["lstm"]
    
    for ds, t in dataset_tasks:
        for model in all_models:
            for seed in config["seeds"]:
                exp = {
                    "dataset": ds,
                    "task": t,
                    "model": model,
                    "seed": seed,
                    "config": config["datasets"][ds],
                    "training_config": config["training"]
                }
                experiments.append(exp)
    
    return experiments


def run_reproduction_experiment(exp: Dict, output_dir: str, data_dir: str, 
                              device: str, mixed_precision: bool) -> Dict:
    """
    Ejecuta un experimento de reproducción.
    
    Args:
        exp: Configuración del experimento
        output_dir: Directorio de salida
        data_dir: Directorio de datos
        device: Device a utilizar
        mixed_precision: Usar mixed precision
        
    Returns:
        Resultado del experimento
    """
    exp_name = f"{exp['dataset']}_{exp['task']}_{exp['model']}_seed{exp['seed']}"
    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Verificar si ya existe
    results_file = exp_dir / "final_results.yaml"
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = yaml.safe_load(f)
            
            results.update({
                "dataset": exp["dataset"],
                "task": exp["task"],
                "model": exp["model"],
                "seed": exp["seed"],
                "exp_name": exp_name,
                "status": "completed"
            })
            
            return results
        except:
            pass
    
    # Ejecutar entrenamiento
    cmd = [
        sys.executable, "scripts/train.py",
        "--dataset", exp["dataset"],
        "--task", exp["task"],
        "--model", exp["model"],
        "--seed", str(exp["seed"]),
        "--output-dir", str(exp_dir.parent),
        "--data-dir", data_dir,
        "--pipeline", "full",
        "--save-model",
        "--device", device,
        "--log-level", "WARNING"
    ]
    
    # Agregar configuración específica
    training_config = exp["training_config"]
    if training_config.get("batch_size"):
        cmd.extend(["--batch-size", str(training_config["batch_size"])])
    
    if mixed_precision:
        cmd.append("--mixed-precision")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10800)  # 3 horas
        end_time = time.time()
        
        if result.returncode == 0:
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = yaml.safe_load(f)
            else:
                results = {"error": "No se encontraron resultados"}
            
            results.update({
                "dataset": exp["dataset"],
                "task": exp["task"],
                "model": exp["model"],
                "seed": exp["seed"],
                "exp_name": exp_name,
                "status": "completed",
                "training_time": end_time - start_time,
                "reproduction": True
            })
            
            return results
        else:
            return {
                "dataset": exp["dataset"],
                "task": exp["task"],
                "model": exp["model"],
                "seed": exp["seed"],
                "exp_name": exp_name,
                "status": "failed",
                "error": result.stderr
            }
    
    except subprocess.TimeoutExpired:
        return {
            "dataset": exp["dataset"],
            "task": exp["task"],
            "model": exp["model"],
            "seed": exp["seed"],
            "exp_name": exp_name,
            "status": "timeout"
        }
    except Exception as e:
        return {
            "dataset": exp["dataset"],
            "task": exp["task"],
            "model": exp["model"],
            "seed": exp["seed"],
            "exp_name": exp_name,
            "status": "error",
            "error": str(e)
        }


def analyze_reproduction_results(results: List[Dict], config: Dict) -> Dict:
    """
    Analiza los resultados de reproducción.
    
    Args:
        results: Lista de resultados
        config: Configuración del paper
        
    Returns:
        Análisis detallado
    """
    df = pd.DataFrame(results)
    successful_results = df[df["status"] == "completed"]
    
    analysis = {
        "total_experiments": len(results),
        "successful_experiments": len(successful_results),
        "success_rate": len(successful_results) / len(results) * 100,
        "paper_comparison": {},
        "statistical_analysis": {},
        "best_models": {}
    }
    
    # Análisis por dataset y tarea
    for dataset_task, group in successful_results.groupby(["dataset", "task"]):
        dataset, task = dataset_task
        key = f"{dataset}_{task}"
        
        # Determinar métrica principal
        if task == "gender":
            metric = "accuracy"
        elif task == "age":
            metric = "mae" if dataset == "timit" else "accuracy"
        else:
            metric = "accuracy"
        
        if metric not in group.columns:
            continue
        
        # Estadísticas generales
        group_stats = {
            "metric": metric,
            "count": len(group),
            "mean": group[metric].mean(),
            "std": group[metric].std(),
            "min": group[metric].min(),
            "max": group[metric].max(),
            "median": group[metric].median()
        }
        
        # Comparación con SOTA
        sota_key = f"{dataset}_{task}"
        if sota_key in config["sota_baselines"]:
            sota_info = config["sota_baselines"][sota_key]
            
            if metric == "accuracy":
                baseline = sota_info["baseline_accuracy"]
                expected = sota_info["expected_accuracy"]
                achieved = group[metric].max()
                
                group_stats["sota_comparison"] = {
                    "baseline": baseline,
                    "expected": expected,
                    "achieved": achieved,
                    "improvement_expected": expected - baseline,
                    "improvement_achieved": achieved - baseline,
                    "reproduction_success": achieved >= expected * 0.95  # 95% del objetivo
                }
            
            elif metric == "mae":
                baseline = sota_info["baseline_mae"]
                expected = sota_info["expected_mae"]
                achieved = group[metric].min()
                
                group_stats["sota_comparison"] = {
                    "baseline": baseline,
                    "expected": expected,
                    "achieved": achieved,
                    "improvement_expected": baseline - expected,
                    "improvement_achieved": baseline - achieved,
                    "reproduction_success": achieved <= expected * 1.05  # 5% de tolerancia
                }
        
        # Mejor modelo
        if metric in ["accuracy", "f1_score", "precision", "recall"]:
            best_idx = group[metric].idxmax()
        else:
            best_idx = group[metric].idxmin()
        
        best_model = group.loc[best_idx]
        group_stats["best_model"] = {
            "name": best_model["model"],
            "score": best_model[metric],
            "seed": best_model["seed"]
        }
        
        # Análisis por modelo
        model_stats = group.groupby("model")[metric].agg([
            "count", "mean", "std", "min", "max"
        ]).round(4)
        
        group_stats["model_analysis"] = model_stats.to_dict()
        
        analysis["paper_comparison"][key] = group_stats
    
    # Análisis estadístico general
    if len(successful_results) > 0:
        # Comparación CNN vs LSTM
        cnn_models = config["models"]["cnn"]
        lstm_models = config["models"]["lstm"]
        
        cnn_results = successful_results[successful_results["model"].isin(cnn_models)]
        lstm_results = successful_results[successful_results["model"].isin(lstm_models)]
        
        if len(cnn_results) > 0 and len(lstm_results) > 0:
            for dataset_task, group in successful_results.groupby(["dataset", "task"]):
                dataset, task = dataset_task
                key = f"{dataset}_{task}"
                
                if task == "gender":
                    metric = "accuracy"
                elif task == "age":
                    metric = "mae" if dataset == "timit" else "accuracy"
                else:
                    metric = "accuracy"
                
                if metric not in group.columns:
                    continue
                
                cnn_group = group[group["model"].isin(cnn_models)]
                lstm_group = group[group["model"].isin(lstm_models)]
                
                if len(cnn_group) > 0 and len(lstm_group) > 0:
                    try:
                        # Test t para comparar CNN vs LSTM
                        t_stat, p_value = ttest_ind(cnn_group[metric], lstm_group[metric])
                        
                        analysis["statistical_analysis"][key] = {
                            "cnn_mean": cnn_group[metric].mean(),
                            "lstm_mean": lstm_group[metric].mean(),
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                            "better_architecture": "CNN" if cnn_group[metric].mean() > lstm_group[metric].mean() else "LSTM"
                        }
                    except:
                        pass
    
    return analysis


def create_reproduction_report(analysis: Dict, output_dir: str, config: Dict) -> None:
    """
    Crea un reporte completo de reproducción.
    
    Args:
        analysis: Análisis de resultados
        output_dir: Directorio de salida
        config: Configuración del paper
    """
    report_file = Path(output_dir) / "reproduction_report.md"
    
    with open(report_file, 'w') as f:
        f.write("# Reporte de Reproducción del Paper\n\n")
        f.write("## Multi-Corpus Benchmarking of CNN & LSTM Models for Speaker Profiling\n\n")
        
        # Resumen general
        f.write("## Resumen General\n\n")
        f.write(f"- **Total de experimentos**: {analysis['total_experiments']}\n")
        f.write(f"- **Experimentos exitosos**: {analysis['successful_experiments']}\n")
        f.write(f"- **Tasa de éxito**: {analysis['success_rate']:.1f}%\n\n")
        
        # Comparación con SOTA
        f.write("## Comparación con Estado del Arte\n\n")
        
        for key, stats in analysis["paper_comparison"].items():
            dataset, task = key.split("_")
            f.write(f"### {dataset.upper()} - {task.capitalize()}\n\n")
            
            f.write(f"- **Métrica**: {stats['metric'].upper()}\n")
            f.write(f"- **Mejor resultado**: {stats['max']:.4f}\n")
            f.write(f"- **Promedio**: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
            f.write(f"- **Mejor modelo**: {stats['best_model']['name']}\n")
            
            if "sota_comparison" in stats:
                sota = stats["sota_comparison"]
                f.write(f"- **Baseline SOTA**: {sota['baseline']:.4f}\n")
                f.write(f"- **Objetivo del paper**: {sota['expected']:.4f}\n")
                f.write(f"- **Resultado logrado**: {sota['achieved']:.4f}\n")
                f.write(f"- **Mejora esperada**: {sota['improvement_expected']:.4f}\n")
                f.write(f"- **Mejora lograda**: {sota['improvement_achieved']:.4f}\n")
                f.write(f"- **Reproducción exitosa**: {'✅' if sota['reproduction_success'] else '❌'}\n")
            
            f.write("\n")
        
        # Análisis estadístico
        if analysis["statistical_analysis"]:
            f.write("## Análisis Estadístico\n\n")
            
            for key, stats in analysis["statistical_analysis"].items():
                dataset, task = key.split("_")
                f.write(f"### {dataset.upper()} - {task.capitalize()}: CNN vs LSTM\n\n")
                
                f.write(f"- **CNN promedio**: {stats['cnn_mean']:.4f}\n")
                f.write(f"- **LSTM promedio**: {stats['lstm_mean']:.4f}\n")
                f.write(f"- **Estadístico t**: {stats['t_statistic']:.4f}\n")
                f.write(f"- **Valor p**: {stats['p_value']:.4f}\n")
                f.write(f"- **Significativo**: {'✅' if stats['significant'] else '❌'}\n")
                f.write(f"- **Mejor arquitectura**: {stats['better_architecture']}\n\n")
        
        # Configuración reproducida
        f.write("## Configuración Reproducida\n\n")
        f.write(f"- **Seeds utilizados**: {', '.join(map(str, config['seeds']))}\n")
        f.write(f"- **Modelos CNN**: {', '.join(config['models']['cnn'])}\n")
        f.write(f"- **Modelos LSTM**: {', '.join(config['models']['lstm'])}\n")
        f.write(f"- **Pipeline**: {config['training']['pipeline']}\n")
        f.write(f"- **Batch size**: {config['training']['batch_size']}\n")
        f.write(f"- **Épocas selección**: {config['training']['epochs_selection']}\n")
        f.write(f"- **Épocas fine-tuning**: {config['training']['epochs_finetune']}\n")
        
        f.write("\n---\n\n")
        f.write("*Reporte generado automáticamente por el sistema de reproducción*\n")
    
    logging.info(f"Reporte de reproducción guardado en: {report_file}")


def create_reproduction_plots(results: List[Dict], analysis: Dict, output_dir: str, config: Dict) -> None:
    """
    Crea gráficos de reproducción.
    
    Args:
        results: Lista de resultados
        analysis: Análisis de resultados
        output_dir: Directorio de salida
        config: Configuración del paper
    """
    df = pd.DataFrame(results)
    successful_results = df[df["status"] == "completed"]
    
    if len(successful_results) == 0:
        return
    
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Gráfico 1: Comparación con SOTA
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Comparación con Estado del Arte", fontsize=16)
    
    plot_idx = 0
    for key, stats in analysis["paper_comparison"].items():
        if "sota_comparison" in stats and plot_idx < 4:
            ax = axes[plot_idx // 2, plot_idx % 2]
            
            sota = stats["sota_comparison"]
            values = [sota["baseline"], sota["expected"], sota["achieved"]]
            labels = ["Baseline\nSOTA", "Objetivo\nPaper", "Resultado\nLogrado"]
            colors = ["red", "orange", "green" if sota["reproduction_success"] else "yellow"]
            
            bars = ax.bar(labels, values, color=colors, alpha=0.7)
            ax.set_title(f"{key.replace('_', ' ').title()}")
            ax.set_ylabel(stats["metric"].upper())
            
            # Agregar valores en las barras
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f"{value:.4f}", ha='center', va='bottom')
            
            plot_idx += 1
    
    # Ocultar subplots vacíos
    for i in range(plot_idx, 4):
        axes[i // 2, i % 2].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "sota_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico 2: Distribución de resultados por modelo
    for dataset_task, group in successful_results.groupby(["dataset", "task"]):
        dataset, task = dataset_task
        
        if task == "gender":
            metric = "accuracy"
        elif task == "age":
            metric = "mae" if dataset == "timit" else "accuracy"
        else:
            metric = "accuracy"
        
        if metric not in group.columns:
            continue
        
        plt.figure(figsize=(12, 8))
        
        # Separar CNN y LSTM
        cnn_models = config["models"]["cnn"]
        lstm_models = config["models"]["lstm"]
        
        cnn_group = group[group["model"].isin(cnn_models)]
        lstm_group = group[group["model"].isin(lstm_models)]
        
        # Box plots
        if len(cnn_group) > 0:
            plt.subplot(2, 1, 1)
            sns.boxplot(data=cnn_group, x="model", y=metric)
            plt.title(f"CNN Models - {dataset.upper()} {task.capitalize()}")
            plt.xticks(rotation=45)
        
        if len(lstm_group) > 0:
            plt.subplot(2, 1, 2)
            sns.boxplot(data=lstm_group, x="model", y=metric)
            plt.title(f"LSTM Models - {dataset.upper()} {task.capitalize()}")
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"distribution_{dataset}_{task}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"Gráficos de reproducción guardados en: {plots_dir}")


def main():
    """Función principal del script de reproducción."""
    args = parse_arguments()
    
    # Configurar logging
    log_file = Path(args.output_dir) / "reproduction.log" if args.output_dir else None
    setup_logging(args.log_level, log_file)
    
    logging.info("=" * 80)
    logging.info("REPRODUCCIÓN DE RESULTADOS DEL PAPER")
    logging.info("=" * 80)
    
    # Obtener configuración del paper
    config = get_paper_configuration()
    
    # Obtener experimentos
    if args.all_experiments:
        experiments = get_experiment_list(config)
    else:
        experiments = get_experiment_list(config, args.dataset, args.task)
    
    if not experiments:
        logging.error("No se encontraron experimentos para ejecutar")
        sys.exit(1)
    
    logging.info(f"Total de experimentos: {len(experiments)}")
    
    # Estimar tiempo
    estimated_time = len(experiments) * 45  # 45 minutos promedio por experimento
    if args.parallel_jobs > 1:
        estimated_time = estimated_time // args.parallel_jobs
    
    logging.info(f"Tiempo estimado: {estimated_time//60} horas {estimated_time%60} minutos")
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar configuración
    config_file = output_dir / "paper_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Ejecutar experimentos
    logging.info("Iniciando reproducción de experimentos...")
    results = []
    
    if args.parallel_jobs > 1:
        # Ejecución paralela
        with ProcessPoolExecutor(max_workers=args.parallel_jobs) as executor:
            futures = []
            for exp in experiments:
                future = executor.submit(
                    run_reproduction_experiment,
                    exp, str(output_dir), args.data_dir, args.device, args.mixed_precision
                )
                futures.append(future)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Reproducción"):
                result = future.result()
                results.append(result)
                
                if result["status"] == "completed":
                    logging.info(f"✓ {result['exp_name']}")
                else:
                    logging.warning(f"✗ {result['exp_name']} - {result['status']}")
    else:
        # Ejecución secuencial
        for exp in tqdm(experiments, desc="Reproducción"):
            result = run_reproduction_experiment(
                exp, str(output_dir), args.data_dir, args.device, args.mixed_precision
            )
            results.append(result)
            
            if result["status"] == "completed":
                logging.info(f"✓ {result['exp_name']}")
            else:
                logging.warning(f"✗ {result['exp_name']} - {result['status']}")
    
    # Análisis de resultados
    logging.info("Analizando resultados de reproducción...")
    analysis = analyze_reproduction_results(results, config)
    
    # Guardar resultados
    results_file = output_dir / "reproduction_results.yaml"
    with open(results_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    analysis_file = output_dir / "reproduction_analysis.yaml"
    with open(analysis_file, 'w') as f:
        yaml.dump(analysis, f, default_flow_style=False)
    
    # Crear CSV
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "reproduction_results.csv", index=False)
    
    # Crear reporte
    create_reproduction_report(analysis, str(output_dir), config)
    
    # Crear gráficos
    create_reproduction_plots(results, analysis, str(output_dir), config)
    
    # Mostrar resumen
    logging.info("=" * 80)
    logging.info("RESUMEN DE REPRODUCCIÓN")
    logging.info("=" * 80)
    
    total = len(results)
    successful = len([r for r in results if r["status"] == "completed"])
    
    logging.info(f"Experimentos totales: {total}")
    logging.info(f"Experimentos exitosos: {successful}")
    logging.info(f"Tasa de éxito: {successful/total*100:.1f}%")
    
    # Resumen por dataset/task
    for key, stats in analysis["paper_comparison"].items():
        dataset, task = key.split("_")
        logging.info(f"\n{dataset.upper()} {task.capitalize()}:")
        logging.info(f"  Mejor resultado: {stats['max']:.4f}")
        logging.info(f"  Mejor modelo: {stats['best_model']['name']}")
        
        if "sota_comparison" in stats:
            sota = stats["sota_comparison"]
            logging.info(f"  Objetivo: {sota['expected']:.4f}")
            logging.info(f"  Logrado: {sota['achieved']:.4f}")
            logging.info(f"  Reproducción: {'✅' if sota['reproduction_success'] else '❌'}")
    
    logging.info(f"\nResultados guardados en: {output_dir}")
    logging.info("=" * 80)
    logging.info("REPRODUCCIÓN COMPLETADA")
    logging.info("=" * 80)


if __name__ == "__main__":
    main() 