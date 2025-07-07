"""
Evaluation Metrics Module
========================

This module implements evaluation metrics for speaker profiling
as specified in "Multi-Corpus Benchmarking of CNN & LSTM Models for Speaker Profiling".

Metrics include:
- Classification: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- Regression: MAE, MSE, RMSE, R²
- Statistical analysis: Confidence intervals, significance tests
- Cross-corpus evaluation

Authors: Jorge Jorrin-Coz et al., 2025
License: MIT
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import bootstrap
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from collections import defaultdict

class MetricsTracker:
    """
    Tracks and manages evaluation metrics across experiments.
    
    Provides functionality to calculate, store, and analyze metrics
    for multiple models, datasets, and experimental conditions.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.results = defaultdict(list)
        self.metadata = {}
    
    def add_result(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        metadata: Dict[str, Any] = None
    ):
        """
        Add experimental result.
        
        Args:
            experiment_id (str): Unique identifier for the experiment
            metrics (Dict[str, float]): Dictionary of metric values
            metadata (Dict[str, Any]): Additional metadata
        """
        result = {
            'experiment_id': experiment_id,
            'metrics': metrics.copy(),
            'metadata': metadata or {}
        }
        
        self.results[experiment_id].append(result)
    
    def get_summary(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for an experiment.
        
        Args:
            experiment_id (str): Experiment identifier
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        if experiment_id not in self.results:
            return {}
        
        results = self.results[experiment_id]
        
        if not results:
            return {}
        
        # Extract all metrics
        all_metrics = {}
        for result in results:
            for metric_name, value in result['metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Calculate statistics
        summary = {}
        for metric_name, values in all_metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'count': len(values),
                    'values': values
                }
        
        return summary
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metric_name: str = 'accuracy'
    ) -> pd.DataFrame:
        """
        Compare multiple experiments on a specific metric.
        
        Args:
            experiment_ids (List[str]): List of experiment IDs to compare
            metric_name (str): Metric to compare
            
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []
        
        for exp_id in experiment_ids:
            summary = self.get_summary(exp_id)
            
            if metric_name in summary:
                stats = summary[metric_name]
                comparison_data.append({
                    'experiment': exp_id,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'median': stats['median'],
                    'count': stats['count']
                })
        
        return pd.DataFrame(comparison_data)

def calculate_classification_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    average: str = 'weighted',
    labels: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true (Union[List, np.ndarray]): True labels
        y_pred (Union[List, np.ndarray]): Predicted labels
        average (str): Averaging strategy for multi-class metrics
        labels (Optional[List[str]]): Label names for reporting
        
    Returns:
        Dict[str, float]: Dictionary of classification metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # Per-class metrics for binary classification
    if len(np.unique(y_true)) == 2:
        metrics.update({
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_score_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_score_micro': f1_score(y_true, y_pred, average='micro', zero_division=0)
        })
    
    return metrics

def calculate_regression_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray]
) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true (Union[List, np.ndarray]): True values
        y_pred (Union[List, np.ndarray]): Predicted values
        
    Returns:
        Dict[str, float]: Dictionary of regression metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2_score': r2_score(y_true, y_pred)
    }
    
    # Additional regression metrics
    residuals = y_true - y_pred
    metrics.update({
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'mean_absolute_percentage_error': np.mean(np.abs(residuals / (y_true + 1e-8))) * 100
    })
    
    return metrics

def calculate_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    task: str = 'classification',
    **kwargs
) -> Dict[str, float]:
    """
    Calculate appropriate metrics based on task type.
    
    Args:
        y_true (Union[List, np.ndarray]): True labels/values
        y_pred (Union[List, np.ndarray]): Predicted labels/values
        task (str): Task type ('classification', 'regression', 'gender', 'age')
        **kwargs: Additional arguments for metric calculation
        
    Returns:
        Dict[str, float]: Dictionary of calculated metrics
    """
    if task in ['classification', 'gender', 'age']:
        return calculate_classification_metrics(y_true, y_pred, **kwargs)
    elif task == 'regression':
        return calculate_regression_metrics(y_true, y_pred)
    else:
        raise ValueError(f"Unknown task type: {task}")

def calculate_confidence_interval(
    values: Union[List, np.ndarray],
    confidence: float = 0.95,
    method: str = 'bootstrap'
) -> Tuple[float, float]:
    """
    Calculate confidence interval for a set of values.
    
    Args:
        values (Union[List, np.ndarray]): Values to calculate CI for
        confidence (float): Confidence level (e.g., 0.95 for 95% CI)
        method (str): Method to use ('bootstrap', 'normal')
        
    Returns:
        Tuple[float, float]: Lower and upper bounds of confidence interval
    """
    values = np.array(values)
    
    if len(values) == 0:
        return (0.0, 0.0)
    
    if method == 'bootstrap':
        # Bootstrap confidence interval
        n_bootstrap = 1000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
    elif method == 'normal':
        # Normal distribution confidence interval
        mean = np.mean(values)
        std_err = stats.sem(values)  # Standard error of the mean
        alpha = 1 - confidence
        
        # t-distribution for small samples
        if len(values) < 30:
            t_critical = stats.t.ppf(1 - alpha / 2, len(values) - 1)
            margin = t_critical * std_err
        else:
            z_critical = stats.norm.ppf(1 - alpha / 2)
            margin = z_critical * std_err
        
        lower = mean - margin
        upper = mean + margin
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return (lower, upper)

def statistical_significance_test(
    values1: Union[List, np.ndarray],
    values2: Union[List, np.ndarray],
    test: str = 'ttest',
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    Perform statistical significance test between two sets of values.
    
    Args:
        values1 (Union[List, np.ndarray]): First set of values
        values2 (Union[List, np.ndarray]): Second set of values
        test (str): Statistical test to use ('ttest', 'wilcoxon', 'mannwhitney')
        alternative (str): Alternative hypothesis ('two-sided', 'less', 'greater')
        
    Returns:
        Dict[str, float]: Test results including p-value and effect size
    """
    values1 = np.array(values1)
    values2 = np.array(values2)
    
    results = {
        'mean1': np.mean(values1),
        'mean2': np.mean(values2),
        'std1': np.std(values1),
        'std2': np.std(values2),
        'n1': len(values1),
        'n2': len(values2)
    }
    
    if test == 'ttest':
        # Independent t-test
        statistic, p_value = stats.ttest_ind(values1, values2, alternative=alternative)
        results['test'] = 'independent_ttest'
        
    elif test == 'paired_ttest':
        # Paired t-test (for same subjects/models)
        if len(values1) != len(values2):
            raise ValueError("Paired t-test requires equal length arrays")
        statistic, p_value = stats.ttest_rel(values1, values2, alternative=alternative)
        results['test'] = 'paired_ttest'
        
    elif test == 'wilcoxon':
        # Wilcoxon signed-rank test (non-parametric paired)
        if len(values1) != len(values2):
            raise ValueError("Wilcoxon test requires equal length arrays")
        statistic, p_value = stats.wilcoxon(values1, values2, alternative=alternative)
        results['test'] = 'wilcoxon'
        
    elif test == 'mannwhitney':
        # Mann-Whitney U test (non-parametric independent)
        statistic, p_value = stats.mannwhitneyu(values1, values2, alternative=alternative)
        results['test'] = 'mannwhitney'
        
    else:
        raise ValueError(f"Unknown test: {test}")
    
    results.update({
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05
    })
    
    # Calculate effect size (Cohen's d for t-tests)
    if test in ['ttest', 'paired_ttest']:
        if test == 'paired_ttest':
            # For paired data, use the standard deviation of differences
            differences = values1 - values2
            cohens_d = np.mean(differences) / np.std(differences)
        else:
            # For independent samples
            pooled_std = np.sqrt(((len(values1) - 1) * np.var(values1) + 
                                 (len(values2) - 1) * np.var(values2)) / 
                                (len(values1) + len(values2) - 2))
            cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std
        
        results['effect_size'] = cohens_d
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            results['effect_size_interpretation'] = 'negligible'
        elif abs(cohens_d) < 0.5:
            results['effect_size_interpretation'] = 'small'
        elif abs(cohens_d) < 0.8:
            results['effect_size_interpretation'] = 'medium'
        else:
            results['effect_size_interpretation'] = 'large'
    
    return results

def evaluate_cross_corpus_transfer(
    source_results: Dict[str, List[float]],
    target_results: Dict[str, List[float]],
    metric_name: str = 'accuracy'
) -> Dict[str, Any]:
    """
    Evaluate cross-corpus transfer learning performance.
    
    Args:
        source_results (Dict[str, List[float]]): Results on source corpus
        target_results (Dict[str, List[float]]): Results on target corpus
        metric_name (str): Metric to evaluate
        
    Returns:
        Dict[str, Any]: Transfer learning evaluation results
    """
    transfer_results = {}
    
    for model_name in source_results.keys():
        if model_name in target_results:
            source_scores = source_results[model_name]
            target_scores = target_results[model_name]
            
            # Calculate transfer performance
            source_mean = np.mean(source_scores)
            target_mean = np.mean(target_scores)
            
            # Transfer ratio (how much performance is retained)
            transfer_ratio = target_mean / source_mean if source_mean > 0 else 0
            
            # Performance drop
            performance_drop = source_mean - target_mean
            
            # Statistical significance of transfer
            sig_test = statistical_significance_test(
                source_scores, target_scores, test='paired_ttest'
            )
            
            transfer_results[model_name] = {
                'source_performance': {
                    'mean': source_mean,
                    'std': np.std(source_scores),
                    'values': source_scores
                },
                'target_performance': {
                    'mean': target_mean,
                    'std': np.std(target_scores),
                    'values': target_scores
                },
                'transfer_ratio': transfer_ratio,
                'performance_drop': performance_drop,
                'significance_test': sig_test
            }
    
    return transfer_results

def create_confusion_matrix_analysis(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create comprehensive confusion matrix analysis.
    
    Args:
        y_true (Union[List, np.ndarray]): True labels
        y_pred (Union[List, np.ndarray]): Predicted labels
        labels (Optional[List[str]]): Label names
        
    Returns:
        Dict[str, Any]: Confusion matrix analysis
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Get unique labels
    unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    
    if labels is None:
        labels = [f"Class_{i}" for i in unique_labels]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for i, label in enumerate(labels):
        if i < len(unique_labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': cm[i, :].sum()
            }
    
    return {
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist(),
        'labels': labels,
        'per_class_metrics': per_class_metrics,
        'overall_accuracy': accuracy_score(y_true, y_pred)
    }

def benchmark_comparison(
    our_results: Dict[str, float],
    baseline_results: Dict[str, float],
    improvement_threshold: float = 0.01
) -> Dict[str, Any]:
    """
    Compare our results with baseline/SOTA results.
    
    Args:
        our_results (Dict[str, float]): Our model results
        baseline_results (Dict[str, float]): Baseline/SOTA results
        improvement_threshold (float): Minimum improvement to consider significant
        
    Returns:
        Dict[str, Any]: Benchmark comparison results
    """
    comparison = {}
    
    for metric_name in our_results.keys():
        if metric_name in baseline_results:
            our_score = our_results[metric_name]
            baseline_score = baseline_results[metric_name]
            
            improvement = our_score - baseline_score
            improvement_pct = (improvement / baseline_score) * 100 if baseline_score > 0 else 0
            
            is_better = improvement > improvement_threshold
            
            comparison[metric_name] = {
                'our_score': our_score,
                'baseline_score': baseline_score,
                'improvement': improvement,
                'improvement_percentage': improvement_pct,
                'is_better': is_better,
                'is_significant': abs(improvement) > improvement_threshold
            }
    
    return comparison

class ResultsAggregator:
    """
    Aggregates and analyzes results from multiple experiments.
    
    Useful for combining results from multiple seeds, models, or datasets.
    """
    
    def __init__(self):
        """Initialize results aggregator."""
        self.experiments = {}
    
    def add_experiment(
        self,
        experiment_name: str,
        results: List[Dict[str, float]],
        metadata: Dict[str, Any] = None
    ):
        """
        Add experiment results.
        
        Args:
            experiment_name (str): Name of the experiment
            results (List[Dict[str, float]]): List of result dictionaries
            metadata (Dict[str, Any]): Experiment metadata
        """
        self.experiments[experiment_name] = {
            'results': results,
            'metadata': metadata or {}
        }
    
    def get_aggregated_stats(self, experiment_name: str) -> Dict[str, Dict[str, float]]:
        """
        Get aggregated statistics for an experiment.
        
        Args:
            experiment_name (str): Name of the experiment
            
        Returns:
            Dict[str, Dict[str, float]]: Aggregated statistics per metric
        """
        if experiment_name not in self.experiments:
            return {}
        
        results = self.experiments[experiment_name]['results']
        
        # Collect all metrics
        all_metrics = defaultdict(list)
        for result in results:
            for metric_name, value in result.items():
                all_metrics[metric_name].append(value)
        
        # Calculate statistics
        stats = {}
        for metric_name, values in all_metrics.items():
            if values:
                ci_lower, ci_upper = calculate_confidence_interval(values)
                
                stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'count': len(values),
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                }
        
        return stats
    
    def compare_experiments(
        self,
        experiment_names: List[str],
        metric_name: str = 'accuracy'
    ) -> pd.DataFrame:
        """
        Compare multiple experiments on a specific metric.
        
        Args:
            experiment_names (List[str]): Names of experiments to compare
            metric_name (str): Metric to compare
            
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []
        
        for exp_name in experiment_names:
            stats = self.get_aggregated_stats(exp_name)
            
            if metric_name in stats:
                metric_stats = stats[metric_name]
                comparison_data.append({
                    'experiment': exp_name,
                    'mean': metric_stats['mean'],
                    'std': metric_stats['std'],
                    'ci_lower': metric_stats['ci_lower'],
                    'ci_upper': metric_stats['ci_upper'],
                    'count': metric_stats['count']
                })
        
        return pd.DataFrame(comparison_data)
    
    def generate_summary_report(self) -> str:
        """
        Generate a summary report of all experiments.
        
        Returns:
            str: Summary report as text
        """
        report_lines = ["EXPERIMENT SUMMARY REPORT", "=" * 50, ""]
        
        for exp_name, exp_data in self.experiments.items():
            report_lines.append(f"Experiment: {exp_name}")
            report_lines.append("-" * 30)
            
            stats = self.get_aggregated_stats(exp_name)
            
            for metric_name, metric_stats in stats.items():
                mean = metric_stats['mean']
                std = metric_stats['std']
                ci_lower = metric_stats['ci_lower']
                ci_upper = metric_stats['ci_upper']
                
                report_lines.append(
                    f"{metric_name:15}: {mean:.4f} ± {std:.4f} "
                    f"(95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])"
                )
            
            report_lines.append("")
        
        return "\n".join(report_lines) 