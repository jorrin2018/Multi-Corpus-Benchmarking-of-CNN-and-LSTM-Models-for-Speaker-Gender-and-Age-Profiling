"""
Training Module
==============

This module implements the two-stage training pipeline for speaker profiling
as specified in "Multi-Corpus Benchmarking of CNN & LSTM Models for Speaker Profiling".

Training Pipeline:
1. Stage 1: Model Selection - Train multiple models with different architectures
2. Stage 2: Fine-tuning - Fine-tune best model with optimized hyperparameters

Features:
- Multi-seed training (10 seeds) for statistical significance
- Early stopping with patience
- Learning rate scheduling
- Gradient clipping
- Mixed precision training (optional)
- Comprehensive logging and checkpointing

Authors: Jorge Jorrin-Coz et al., 2025
License: MIT
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from pathlib import Path
import time
import json
from collections import defaultdict
import random

from ..models.cnn_models import create_cnn_model
from ..models.lstm_models import create_lstm_model
from .callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from ..evaluation.metrics import calculate_metrics, MetricsTracker

class SpeakerProfilingTrainer:
    """
    Trainer for speaker profiling models with two-stage pipeline.
    
    Implements the training methodology specified in the paper:
    - Stage 1: Model selection across architectures
    - Stage 2: Fine-tuning of best model
    - Multi-seed training for statistical significance
    """
    
    def __init__(
        self,
        output_dir: str,
        device: str = 'auto',
        mixed_precision: bool = True,
        log_interval: int = 10,
        save_best_only: bool = True,
        verbose: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            output_dir (str): Directory to save results and checkpoints
            device (str): Device to use ('auto', 'cuda', 'cpu')
            mixed_precision (bool): Use mixed precision training
            log_interval (int): Logging interval in steps
            save_best_only (bool): Only save best model checkpoints
            verbose (bool): Verbose logging
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.mixed_precision = mixed_precision and self.device.type == 'cuda'
        self.log_interval = log_interval
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        # Initialize scaler for mixed precision
        if self.mixed_precision:
            self.scaler = GradScaler()
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        
        # Results storage
        self.results = defaultdict(list)
        
        if self.verbose:
            print(f"Trainer initialized on device: {self.device}")
            print(f"Mixed precision: {self.mixed_precision}")
            print(f"Output directory: {self.output_dir}")
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def create_model(
        self,
        architecture: str,
        model_type: str,
        num_classes: int,
        input_dim: Optional[int] = None,
        **model_kwargs
    ) -> nn.Module:
        """
        Create model based on architecture and type.
        
        Args:
            architecture (str): Model architecture name
            model_type (str): Type of model ('cnn', 'lstm')
            num_classes (int): Number of output classes
            input_dim (Optional[int]): Input dimension for LSTM models
            **model_kwargs: Additional model arguments
            
        Returns:
            nn.Module: Initialized model
        """
        if model_type == 'cnn':
            model = create_cnn_model(
                architecture=architecture,
                num_classes=num_classes,
                **model_kwargs
            )
        elif model_type == 'lstm':
            if input_dim is None:
                raise ValueError("input_dim required for LSTM models")
            model = create_lstm_model(
                input_dim=input_dim,
                num_classes=num_classes,
                **model_kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model.to(self.device)
    
    def create_optimizer(
        self,
        model: nn.Module,
        optimizer_name: str = 'adam',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        **optimizer_kwargs
    ) -> optim.Optimizer:
        """
        Create optimizer for model.
        
        Args:
            model (nn.Module): Model to optimize
            optimizer_name (str): Optimizer name ('adam', 'sgd', 'adamw')
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay
            **optimizer_kwargs: Additional optimizer arguments
            
        Returns:
            optim.Optimizer: Configured optimizer
        """
        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **optimizer_kwargs
            )
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9,
                **optimizer_kwargs
            )
        elif optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **optimizer_kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return optimizer
    
    def create_loss_function(
        self,
        task: str,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """
        Create loss function based on task.
        
        Args:
            task (str): Task type ('gender', 'age', 'regression')
            num_classes (int): Number of classes
            class_weights (Optional[torch.Tensor]): Class weights for imbalanced data
            
        Returns:
            nn.Module: Loss function
        """
        if task in ['gender', 'age'] and num_classes > 1:
            # Classification task
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        elif task == 'regression' or num_classes == 1:
            # Regression task
            loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown task: {task}")
        
        return loss_fn.to(self.device)
    
    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        epoch: int,
        max_grad_norm: float = 1.0
    ) -> Dict[str, float]:
        """
        Train model for one epoch.
        
        Args:
            model (nn.Module): Model to train
            train_loader (DataLoader): Training data loader
            optimizer (optim.Optimizer): Optimizer
            loss_fn (nn.Module): Loss function
            epoch (int): Current epoch number
            max_grad_norm (float): Maximum gradient norm for clipping
            
        Returns:
            Dict[str, float]: Training metrics
        """
        model.train()
        
        total_loss = 0.0
        total_samples = 0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Handle target shape for different tasks
            if targets.dim() > 1 and targets.shape[1] == 1:
                targets = targets.squeeze(1)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.mixed_precision:
                with autocast():
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if max_grad_norm > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Optimizer step
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Optimizer step
                optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            # Logging
            if batch_idx % self.log_interval == 0 and self.verbose:
                current_loss = total_loss / total_samples
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:3d} | Batch {batch_idx:4d}/{len(train_loader):4d} | "
                      f"Loss: {current_loss:.4f} | Time: {elapsed:.1f}s")
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        
        return {
            'loss': avg_loss,
            'samples': total_samples,
            'time': time.time() - start_time
        }
    
    def validate_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        task: str
    ) -> Dict[str, float]:
        """
        Validate model for one epoch.
        
        Args:
            model (nn.Module): Model to validate
            val_loader (DataLoader): Validation data loader
            loss_fn (nn.Module): Loss function
            task (str): Task type for metrics calculation
            
        Returns:
            Dict[str, float]: Validation metrics
        """
        model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Handle target shape
                if targets.dim() > 1 and targets.shape[1] == 1:
                    targets = targets.squeeze(1)
                
                # Forward pass
                if self.mixed_precision:
                    with autocast():
                        outputs = model(inputs)
                        loss = loss_fn(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                
                # Update metrics
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
                
                # Store predictions and targets
                if task == 'regression':
                    all_predictions.extend(outputs.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / total_samples
        
        # Calculate task-specific metrics
        metrics = calculate_metrics(
            y_true=all_targets,
            y_pred=all_predictions,
            task=task
        )
        
        metrics['loss'] = avg_loss
        return metrics
    
    def train_single_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        seed: int,
        experiment_name: str
    ) -> Dict[str, Any]:
        """
        Train a single model with given configuration.
        
        Args:
            model (nn.Module): Model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            test_loader (DataLoader): Test data loader
            config (Dict[str, Any]): Training configuration
            seed (int): Random seed
            experiment_name (str): Name of the experiment
            
        Returns:
            Dict[str, Any]: Training results
        """
        # Set seed
        self.set_seed(seed)
        
        # Create optimizer
        optimizer = self.create_optimizer(
            model=model,
            optimizer_name=config.get('optimizer', 'adam'),
            learning_rate=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Create loss function
        loss_fn = self.create_loss_function(
            task=config['task'],
            num_classes=config['num_classes'],
            class_weights=config.get('class_weights')
        )
        
        # Create callbacks
        callbacks = self._create_callbacks(config, experiment_name, seed)
        
        # Training loop
        best_val_metric = float('-inf') if config.get('monitor_mode', 'max') == 'max' else float('inf')
        best_epoch = 0
        train_history = []
        val_history = []
        
        for epoch in range(config.get('epochs', 100)):
            # Train epoch
            train_metrics = self.train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epoch=epoch,
                max_grad_norm=config.get('max_grad_norm', 1.0)
            )
            
            # Validate epoch
            val_metrics = self.validate_epoch(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                task=config['task']
            )
            
            # Store history
            train_history.append(train_metrics)
            val_history.append(val_metrics)
            
            # Check improvement
            current_metric = val_metrics.get(config.get('monitor_metric', 'accuracy'), 0)
            is_better = (
                current_metric > best_val_metric if config.get('monitor_mode', 'max') == 'max'
                else current_metric < best_val_metric
            )
            
            if is_better:
                best_val_metric = current_metric
                best_epoch = epoch
                
                # Save best model
                if self.save_best_only:
                    checkpoint_path = self.output_dir / f"{experiment_name}_seed{seed}_best.pth"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'val_metric': current_metric,
                        'config': config
                    }, checkpoint_path)
            
            # Apply callbacks
            stop_training = False
            for callback in callbacks:
                callback.on_epoch_end(epoch, val_metrics)
                if hasattr(callback, 'early_stop') and callback.early_stop:
                    stop_training = True
                    break
            
            # Learning rate scheduling
            if 'lr_scheduler' in config:
                scheduler = config['lr_scheduler']
                if scheduler == 'reduce_on_plateau':
                    optimizer.param_groups[0]['lr'] *= 0.5
                elif scheduler == 'cosine':
                    lr = config.get('learning_rate', 1e-3) * 0.5 * (1 + np.cos(np.pi * epoch / config.get('epochs', 100)))
                    optimizer.param_groups[0]['lr'] = lr
            
            # Verbose logging
            if self.verbose:
                print(f"Epoch {epoch:3d} | Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Val Metric: {current_metric:.4f} | "
                      f"Best: {best_val_metric:.4f} (Epoch {best_epoch})")
            
            if stop_training:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        # Final evaluation on test set
        if test_loader is not None:
            # Load best model
            if self.save_best_only:
                checkpoint_path = self.output_dir / f"{experiment_name}_seed{seed}_best.pth"
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    model.load_state_dict(checkpoint['model_state_dict'])
            
            test_metrics = self.validate_epoch(
                model=model,
                val_loader=test_loader,
                loss_fn=loss_fn,
                task=config['task']
            )
        else:
            test_metrics = {}
        
        return {
            'experiment_name': experiment_name,
            'seed': seed,
            'best_epoch': best_epoch,
            'best_val_metric': best_val_metric,
            'train_history': train_history,
            'val_history': val_history,
            'test_metrics': test_metrics,
            'config': config
        }
    
    def _create_callbacks(
        self,
        config: Dict[str, Any],
        experiment_name: str,
        seed: int
    ) -> List[Any]:
        """Create callbacks for training."""
        callbacks = []
        
        # Early stopping
        if config.get('early_stopping', True):
            early_stopping = EarlyStopping(
                patience=config.get('patience', 10),
                monitor=config.get('monitor_metric', 'accuracy'),
                mode=config.get('monitor_mode', 'max'),
                verbose=self.verbose
            )
            callbacks.append(early_stopping)
        
        return callbacks
    
    def stage1_model_selection(
        self,
        architectures: List[str],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        seeds: List[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    ) -> Dict[str, Any]:
        """
        Stage 1: Model selection across architectures.
        
        Args:
            architectures (List[str]): List of architecture names to evaluate
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            test_loader (DataLoader): Test data loader
            config (Dict[str, Any]): Training configuration
            seeds (List[int]): List of random seeds
            
        Returns:
            Dict[str, Any]: Results from model selection
        """
        if self.verbose:
            print("=" * 60)
            print("STAGE 1: MODEL SELECTION")
            print("=" * 60)
        
        results = defaultdict(list)
        
        for architecture in architectures:
            if self.verbose:
                print(f"\nEvaluating architecture: {architecture}")
            
            arch_results = []
            
            for seed in seeds:
                if self.verbose:
                    print(f"  Seed {seed}...")
                
                # Create model
                model = self.create_model(
                    architecture=architecture,
                    model_type=config['model_type'],
                    num_classes=config['num_classes'],
                    input_dim=config.get('input_dim'),
                    **config.get('model_kwargs', {})
                )
                
                # Train model
                experiment_name = f"stage1_{architecture}"
                result = self.train_single_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    config=config,
                    seed=seed,
                    experiment_name=experiment_name
                )
                
                arch_results.append(result)
                results[architecture].append(result)
            
            # Calculate statistics for this architecture
            if arch_results:
                test_metrics = [r['test_metrics'] for r in arch_results if r['test_metrics']]
                if test_metrics:
                    metric_name = config.get('monitor_metric', 'accuracy')
                    scores = [m.get(metric_name, 0) for m in test_metrics]
                    
                    if self.verbose:
                        mean_score = np.mean(scores)
                        std_score = np.std(scores)
                        print(f"  {architecture}: {mean_score:.4f} ± {std_score:.4f}")
        
        # Find best architecture
        best_architecture = None
        best_score = float('-inf') if config.get('monitor_mode', 'max') == 'max' else float('inf')
        
        for architecture, arch_results in results.items():
            test_metrics = [r['test_metrics'] for r in arch_results if r['test_metrics']]
            if test_metrics:
                metric_name = config.get('monitor_metric', 'accuracy')
                scores = [m.get(metric_name, 0) for m in test_metrics]
                mean_score = np.mean(scores)
                
                is_better = (
                    mean_score > best_score if config.get('monitor_mode', 'max') == 'max'
                    else mean_score < best_score
                )
                
                if is_better:
                    best_score = mean_score
                    best_architecture = architecture
        
        if self.verbose:
            print(f"\nBest architecture: {best_architecture} (score: {best_score:.4f})")
        
        return {
            'results': dict(results),
            'best_architecture': best_architecture,
            'best_score': best_score
        }
    
    def stage2_fine_tuning(
        self,
        architecture: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        seeds: List[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    ) -> Dict[str, Any]:
        """
        Stage 2: Fine-tuning of best model.
        
        Args:
            architecture (str): Best architecture from stage 1
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            test_loader (DataLoader): Test data loader
            config (Dict[str, Any]): Fine-tuning configuration
            seeds (List[int]): List of random seeds
            
        Returns:
            Dict[str, Any]: Results from fine-tuning
        """
        if self.verbose:
            print("=" * 60)
            print("STAGE 2: FINE-TUNING")
            print("=" * 60)
            print(f"Fine-tuning architecture: {architecture}")
        
        results = []
        
        for seed in seeds:
            if self.verbose:
                print(f"  Seed {seed}...")
            
            # Create model
            model = self.create_model(
                architecture=architecture,
                model_type=config['model_type'],
                num_classes=config['num_classes'],
                input_dim=config.get('input_dim'),
                **config.get('model_kwargs', {})
            )
            
            # Unfreeze more layers for fine-tuning
            if hasattr(model, 'unfreeze_last_layers'):
                model.unfreeze_last_layers(config.get('unfreeze_layers', 4))
            
            # Train model
            experiment_name = f"stage2_{architecture}"
            result = self.train_single_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config=config,
                seed=seed,
                experiment_name=experiment_name
            )
            
            results.append(result)
        
        # Calculate final statistics
        test_metrics = [r['test_metrics'] for r in results if r['test_metrics']]
        if test_metrics:
            metric_name = config.get('monitor_metric', 'accuracy')
            scores = [m.get(metric_name, 0) for m in test_metrics]
            
            final_stats = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'scores': scores
            }
            
            if self.verbose:
                print(f"\nFinal Results:")
                print(f"  {metric_name}: {final_stats['mean']:.4f} ± {final_stats['std']:.4f}")
                print(f"  Range: [{final_stats['min']:.4f}, {final_stats['max']:.4f}]")
        else:
            final_stats = {}
        
        return {
            'results': results,
            'architecture': architecture,
            'final_stats': final_stats
        }
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON file."""
        output_path = self.output_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        if self.verbose:
            print(f"Results saved to: {output_path}")
    
    def run_full_pipeline(
        self,
        stage1_architectures: List[str],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        stage1_config: Dict[str, Any],
        stage2_config: Dict[str, Any],
        seeds: List[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    ) -> Dict[str, Any]:
        """
        Run the complete two-stage pipeline.
        
        Args:
            stage1_architectures (List[str]): Architectures for stage 1
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            test_loader (DataLoader): Test data loader
            stage1_config (Dict[str, Any]): Stage 1 configuration
            stage2_config (Dict[str, Any]): Stage 2 configuration
            seeds (List[int]): List of random seeds
            
        Returns:
            Dict[str, Any]: Complete pipeline results
        """
        if self.verbose:
            print("Starting two-stage training pipeline...")
        
        # Stage 1: Model selection
        stage1_results = self.stage1_model_selection(
            architectures=stage1_architectures,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=stage1_config,
            seeds=seeds
        )
        
        # Stage 2: Fine-tuning
        best_architecture = stage1_results['best_architecture']
        stage2_results = self.stage2_fine_tuning(
            architecture=best_architecture,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=stage2_config,
            seeds=seeds
        )
        
        # Combine results
        final_results = {
            'stage1': stage1_results,
            'stage2': stage2_results,
            'best_architecture': best_architecture,
            'seeds': seeds,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save results
        self.save_results(final_results, 'training_results.json')
        
        return final_results 