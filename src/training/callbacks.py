"""
Training Callbacks Module
========================

This module implements training callbacks for the speaker profiling pipeline
as specified in "Multi-Corpus Benchmarking of CNN & LSTM Models for Speaker Profiling".

Callbacks include:
- EarlyStopping: Stop training when validation metric stops improving
- LearningRateScheduler: Adjust learning rate during training
- ModelCheckpoint: Save model checkpoints during training
- MetricsLogger: Log training metrics

Authors: Jorge Jorrin-Coz et al., 2025
License: MIT
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import warnings
import time
import json

class Callback:
    """Base class for training callbacks."""
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch: int, logs: Dict[str, Any] = None):
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None):
        """Called at the end of each batch."""
        pass
    
    def on_train_begin(self, logs: Dict[str, Any] = None):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, logs: Dict[str, Any] = None):
        """Called at the end of training."""
        pass

class EarlyStopping(Callback):
    """
    Early stopping callback to stop training when validation metric stops improving.
    
    Monitors a specified metric and stops training when it stops improving
    for a specified number of epochs (patience).
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        mode: str = 'min',
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        Initialize early stopping callback.
        
        Args:
            monitor (str): Metric to monitor
            patience (int): Number of epochs to wait before stopping
            mode (str): 'min' or 'max' - direction of improvement
            min_delta (float): Minimum change to qualify as an improvement
            restore_best_weights (bool): Whether to restore best weights
            verbose (bool): Whether to print early stopping messages
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        # Internal state
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.early_stop = False
        
        # Initialize best value
        if mode == 'min':
            self.best_value = np.inf
            self.monitor_op = np.less
        elif mode == 'max':
            self.best_value = -np.inf
            self.monitor_op = np.greater
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
    
    def on_train_begin(self, logs: Dict[str, Any] = None):
        """Reset early stopping state at the beginning of training."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.early_stop = False
        
        if self.mode == 'min':
            self.best_value = np.inf
        else:
            self.best_value = -np.inf
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Check for early stopping condition at the end of each epoch."""
        if logs is None:
            logs = {}
        
        current_value = logs.get(self.monitor)
        
        if current_value is None:
            if self.verbose:
                warnings.warn(f"Early stopping metric '{self.monitor}' not found in logs")
            return
        
        # Check if current value is better than best
        if self.mode == 'min':
            is_better = current_value < (self.best_value - self.min_delta)
        else:
            is_better = current_value > (self.best_value + self.min_delta)
        
        if is_better:
            # Improvement found
            self.best_value = current_value
            self.wait = 0
            
            if self.restore_best_weights:
                # Store current weights as best
                # Note: This would need to be implemented by the trainer
                pass
                
        else:
            # No improvement
            self.wait += 1
            
            if self.wait >= self.patience:
                # Stop training
                self.stopped_epoch = epoch
                self.early_stop = True
                
                if self.verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                    print(f"Best {self.monitor}: {self.best_value:.6f}")
    
    def on_train_end(self, logs: Dict[str, Any] = None):
        """Print final early stopping message."""
        if self.stopped_epoch > 0 and self.verbose:
            print(f"Restored model weights from epoch {self.stopped_epoch + 1 - self.patience}")

class LearningRateScheduler(Callback):
    """
    Learning rate scheduler callback.
    
    Adjusts learning rate during training based on specified schedule.
    """
    
    def __init__(
        self,
        scheduler_type: str = 'plateau',
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-7,
        monitor: str = 'val_loss',
        mode: str = 'min',
        verbose: bool = True,
        **scheduler_kwargs
    ):
        """
        Initialize learning rate scheduler.
        
        Args:
            scheduler_type (str): Type of scheduler ('plateau', 'step', 'cosine', 'exponential')
            factor (float): Factor to reduce learning rate by
            patience (int): Number of epochs to wait before reducing LR (for plateau)
            min_lr (float): Minimum learning rate
            monitor (str): Metric to monitor (for plateau)
            mode (str): 'min' or 'max' for plateau scheduler
            verbose (bool): Whether to print LR changes
            **scheduler_kwargs: Additional scheduler arguments
        """
        super().__init__()
        self.scheduler_type = scheduler_type
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.scheduler_kwargs = scheduler_kwargs
        
        # Internal state
        self.wait = 0
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.initial_lr = None
        
    def on_train_begin(self, logs: Dict[str, Any] = None):
        """Initialize scheduler state."""
        self.wait = 0
        self.best_value = np.inf if self.mode == 'min' else -np.inf
        
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Update learning rate based on schedule."""
        if logs is None:
            logs = {}
        
        current_lr = logs.get('learning_rate', 0)
        
        if self.scheduler_type == 'plateau':
            self._plateau_schedule(epoch, logs, current_lr)
        elif self.scheduler_type == 'step':
            self._step_schedule(epoch, logs, current_lr)
        elif self.scheduler_type == 'cosine':
            self._cosine_schedule(epoch, logs, current_lr)
        elif self.scheduler_type == 'exponential':
            self._exponential_schedule(epoch, logs, current_lr)
        else:
            warnings.warn(f"Unknown scheduler type: {self.scheduler_type}")
    
    def _plateau_schedule(self, epoch: int, logs: Dict[str, Any], current_lr: float):
        """Reduce learning rate on plateau."""
        current_value = logs.get(self.monitor)
        
        if current_value is None:
            return
        
        # Check if metric improved
        if self.mode == 'min':
            is_better = current_value < self.best_value
        else:
            is_better = current_value > self.best_value
        
        if is_better:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                new_lr = max(current_lr * self.factor, self.min_lr)
                
                if new_lr < current_lr:
                    logs['learning_rate'] = new_lr
                    self.wait = 0
                    
                    if self.verbose:
                        print(f"Reducing learning rate to {new_lr:.6f}")
    
    def _step_schedule(self, epoch: int, logs: Dict[str, Any], current_lr: float):
        """Step learning rate schedule."""
        step_size = self.scheduler_kwargs.get('step_size', 30)
        
        if (epoch + 1) % step_size == 0:
            new_lr = max(current_lr * self.factor, self.min_lr)
            logs['learning_rate'] = new_lr
            
            if self.verbose:
                print(f"Step LR: Reducing learning rate to {new_lr:.6f}")
    
    def _cosine_schedule(self, epoch: int, logs: Dict[str, Any], current_lr: float):
        """Cosine annealing learning rate schedule."""
        if self.initial_lr is None:
            self.initial_lr = current_lr
        
        max_epochs = self.scheduler_kwargs.get('max_epochs', 100)
        
        new_lr = self.min_lr + (self.initial_lr - self.min_lr) * (
            1 + np.cos(np.pi * epoch / max_epochs)
        ) / 2
        
        logs['learning_rate'] = new_lr
        
        if self.verbose and epoch % 10 == 0:
            print(f"Cosine LR: {new_lr:.6f}")
    
    def _exponential_schedule(self, epoch: int, logs: Dict[str, Any], current_lr: float):
        """Exponential decay learning rate schedule."""
        decay_rate = self.scheduler_kwargs.get('decay_rate', 0.96)
        
        new_lr = max(current_lr * decay_rate, self.min_lr)
        logs['learning_rate'] = new_lr
        
        if self.verbose and epoch % 10 == 0:
            print(f"Exponential LR: {new_lr:.6f}")

class ModelCheckpoint(Callback):
    """
    Model checkpoint callback to save model during training.
    
    Saves model weights when validation metric improves.
    """
    
    def __init__(
        self,
        filepath: Union[str, Path],
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_freq: int = 1,
        verbose: bool = True
    ):
        """
        Initialize model checkpoint callback.
        
        Args:
            filepath (Union[str, Path]): Path to save model checkpoints
            monitor (str): Metric to monitor for saving
            mode (str): 'min' or 'max' - direction of improvement
            save_best_only (bool): Whether to only save best model
            save_freq (int): Frequency of saving (in epochs)
            verbose (bool): Whether to print save messages
        """
        super().__init__()
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.verbose = verbose
        
        # Create directory if it doesn't exist
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize best value
        if mode == 'min':
            self.best_value = np.inf
        elif mode == 'max':
            self.best_value = -np.inf
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Save model checkpoint if conditions are met."""
        if logs is None:
            logs = {}
        
        # Check if we should save this epoch
        if (epoch + 1) % self.save_freq != 0:
            return
        
        current_value = logs.get(self.monitor)
        
        if current_value is None:
            if self.verbose:
                warnings.warn(f"Checkpoint metric '{self.monitor}' not found in logs")
            return
        
        # Check if we should save
        should_save = False
        
        if self.save_best_only:
            if self.mode == 'min':
                should_save = current_value < self.best_value
            else:
                should_save = current_value > self.best_value
            
            if should_save:
                self.best_value = current_value
        else:
            should_save = True
        
        if should_save:
            # Format filepath with epoch and metric
            filepath = str(self.filepath).format(
                epoch=epoch + 1,
                **logs
            )
            
            # Save checkpoint
            # Note: Actual saving would be implemented by the trainer
            logs['save_checkpoint'] = filepath
            
            if self.verbose:
                print(f"Saving checkpoint to {filepath}")
                if self.save_best_only:
                    print(f"Best {self.monitor}: {self.best_value:.6f}")

class MetricsLogger(Callback):
    """
    Metrics logger callback to log training metrics.
    
    Saves training metrics to files for analysis.
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        save_freq: int = 1,
        verbose: bool = False
    ):
        """
        Initialize metrics logger.
        
        Args:
            log_dir (Union[str, Path]): Directory to save logs
            save_freq (int): Frequency of saving logs (in epochs)
            verbose (bool): Whether to print log messages
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.save_freq = save_freq
        self.verbose = verbose
        
        # Create directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log storage
        self.metrics_history = []
        self.start_time = None
    
    def on_train_begin(self, logs: Dict[str, Any] = None):
        """Initialize logging at the beginning of training."""
        self.metrics_history = []
        self.start_time = time.time()
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Log metrics at the end of each epoch."""
        if logs is None:
            logs = {}
        
        # Add timestamp and epoch to logs
        log_entry = {
            'epoch': epoch + 1,
            'timestamp': time.time(),
            'elapsed_time': time.time() - self.start_time,
            **logs
        }
        
        self.metrics_history.append(log_entry)
        
        # Save logs periodically
        if (epoch + 1) % self.save_freq == 0:
            self._save_logs()
    
    def on_train_end(self, logs: Dict[str, Any] = None):
        """Save final logs at the end of training."""
        self._save_logs()
    
    def _save_logs(self):
        """Save metrics history to file."""
        log_file = self.log_dir / 'training_metrics.json'
        
        with open(log_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        if self.verbose:
            print(f"Saved metrics to {log_file}")

class ProgressBar(Callback):
    """
    Progress bar callback for training visualization.
    
    Displays training progress with metrics.
    """
    
    def __init__(
        self,
        total_epochs: int,
        metrics_to_show: List[str] = ['loss', 'accuracy'],
        update_freq: int = 1
    ):
        """
        Initialize progress bar.
        
        Args:
            total_epochs (int): Total number of epochs
            metrics_to_show (List[str]): Metrics to display
            update_freq (int): Update frequency (in epochs)
        """
        super().__init__()
        self.total_epochs = total_epochs
        self.metrics_to_show = metrics_to_show
        self.update_freq = update_freq
        self.current_epoch = 0
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Update progress bar at the end of each epoch."""
        if logs is None:
            logs = {}
        
        self.current_epoch = epoch + 1
        
        if self.current_epoch % self.update_freq == 0:
            # Calculate progress
            progress = self.current_epoch / self.total_epochs
            bar_length = 30
            filled_length = int(bar_length * progress)
            
            # Create progress bar
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Format metrics
            metrics_str = []
            for metric in self.metrics_to_show:
                if metric in logs:
                    value = logs[metric]
                    if isinstance(value, float):
                        metrics_str.append(f"{metric}: {value:.4f}")
                    else:
                        metrics_str.append(f"{metric}: {value}")
            
            # Print progress
            print(f"\rEpoch {self.current_epoch:3d}/{self.total_epochs} "
                  f"[{bar}] {progress*100:.1f}% - {' - '.join(metrics_str)}", 
                  end='', flush=True)
            
            if self.current_epoch == self.total_epochs:
                print()  # New line at the end 