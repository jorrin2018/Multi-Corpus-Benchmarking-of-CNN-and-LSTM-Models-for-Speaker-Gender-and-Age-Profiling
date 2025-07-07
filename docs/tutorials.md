# 📚 Tutoriales Paso a Paso - Speaker Profiling

Tutoriales detallados para usar el sistema de **Multi-Corpus Benchmarking para Speaker Profiling** en diferentes escenarios.

## 📋 Índice de Tutoriales

1. [Tutorial 1: Primera Clasificación de Género](#tutorial-1-primera-clasificación-de-género)
2. [Tutorial 2: Estimación de Edad con Regresión](#tutorial-2-estimación-de-edad-con-regresión)
3. [Tutorial 3: Comparación de Modelos CNN vs LSTM](#tutorial-3-comparación-de-modelos-cnn-vs-lstm)
4. [Tutorial 4: Evaluación Cross-Corpus](#tutorial-4-evaluación-cross-corpus)
5. [Tutorial 5: Optimización de Hiperparámetros](#tutorial-5-optimización-de-hiperparámetros)
6. [Tutorial 6: Pipeline Personalizado](#tutorial-6-pipeline-personalizado)
7. [Tutorial 7: Reproducción de Resultados del Paper](#tutorial-7-reproducción-de-resultados-del-paper)

---

## Tutorial 1: Primera Clasificación de Género

**Objetivo**: Entrenar tu primer modelo para clasificación de género usando VoxCeleb1.

### Paso 1: Verificar Instalación

```bash
# Verificar que el sistema esté instalado correctamente
python -c "from src.utils.config_utils import ConfigManager; print('✅ Sistema listo')"

# Verificar estructura de datos (opcional si ya tienes VoxCeleb1)
ls -la data/voxceleb1/
```

### Paso 2: Entrenar Modelo Básico

```bash
# Entrenar MobileNetV2 para clasificación de género
python scripts/train.py \
  --dataset voxceleb1 \
  --task gender \
  --model mobilenetv2 \
  --seed 42 \
  --epochs 20 \
  --batch-size 32 \
  --save-model

# Monitorear progreso en otra terminal
tail -f results/voxceleb1_gender_mobilenetv2_seed42/train.log
```

**Salida esperada:**
```
================================================================================
ENTRENAMIENTO DE SPEAKER PROFILING
================================================================================
Dataset: voxceleb1 (1,251 speakers)
Tarea: gender (2 clases)
Modelo: mobilenetv2 (2.2M parámetros)
Device: cuda

Época 1/20: train_loss=0.652, val_loss=0.543, val_acc=0.734
Época 2/20: train_loss=0.421, val_loss=0.389, val_acc=0.823
...
Época 20/20: train_loss=0.089, val_loss=0.156, val_acc=0.956

✅ Entrenamiento completado en 45 minutos
Mejor modelo guardado en: results/voxceleb1_gender_mobilenetv2_seed42/best_model.pth
```

### Paso 3: Evaluar Modelo

```bash
# Evaluar en conjunto de test
python scripts/evaluate.py \
  --model-path results/voxceleb1_gender_mobilenetv2_seed42/best_model.pth \
  --dataset voxceleb1 \
  --task gender \
  --save-plots \
  --save-predictions
```

**Resultado esperado:**
- Accuracy: ~95-96%
- F1-Score: ~95-96%
- Archivos generados: matriz de confusión, curva ROC, predicciones

### Paso 4: Hacer Predicciones

```bash
# Predecir género en un archivo de audio
python scripts/predict.py \
  --model-path results/voxceleb1_gender_mobilenetv2_seed42/best_model.pth \
  --audio-file sample_audio.wav \
  --task gender \
  --output-probs
```

### ✅ Resultado del Tutorial 1

Has completado exitosamente:
- ✅ Entrenamiento de modelo CNN
- ✅ Evaluación de rendimiento 
- ✅ Predicción en nuevos audios
- ✅ Comprensión del pipeline básico

---

## Tutorial 2: Estimación de Edad con Regresión

**Objetivo**: Implementar regresión de edad usando TIMIT dataset.

### Paso 1: Configurar Dataset TIMIT

```bash
# Verificar estructura de TIMIT
ls -la data/timit/

# Si necesitas configurar splits
python -c "
from src.datasets.timit import TIMITDataset
dataset = TIMITDataset('data/timit', split='train', task='age')
dataset.create_splits()
print('✅ Splits de TIMIT creados')
"
```

### Paso 2: Entrenar para Regresión

```bash
# Entrenar ResNet18 para regresión de edad
python scripts/train.py \
  --dataset timit \
  --task age \
  --model resnet18 \
  --seed 42 \
  --epochs 30 \
  --learning-rate 0.0005 \
  --batch-size 16 \
  --save-model
```

### Paso 3: Analizar Resultados de Regresión

```bash
# Evaluar con métricas de regresión
python scripts/evaluate.py \
  --model-path results/timit_age_resnet18_seed42/best_model.pth \
  --dataset timit \
  --task age \
  --regression-analysis \
  --save-plots
```

**Métricas esperadas:**
- MAE: ~8-12 años
- RMSE: ~12-15 años
- R²: ~0.6-0.7

### Paso 4: Visualizar Predicciones

```python
# analizar_resultados.py
import matplotlib.pyplot as plt
import pandas as pd

# Cargar predicciones
results = pd.read_csv('evaluation_results/predictions.csv')

# Gráfico scatter: edad real vs predicha
plt.figure(figsize=(10, 8))
plt.scatter(results['true_age'], results['predicted_age'], alpha=0.6)
plt.plot([20, 70], [20, 70], 'r--', label='Predicción perfecta')
plt.xlabel('Edad Real')
plt.ylabel('Edad Predicha')
plt.title('Regresión de Edad - Resultados')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Estadísticas por rango de edad
results['age_group'] = pd.cut(results['true_age'], bins=[0, 30, 50, 100], labels=['Joven', 'Adulto', 'Mayor'])
mae_by_group = results.groupby('age_group')['abs_error'].mean()
print("MAE por grupo de edad:")
print(mae_by_group)
```

### ✅ Resultado del Tutorial 2

Has aprendido:
- ✅ Configuración para tareas de regresión
- ✅ Métricas específicas de regresión (MAE, RMSE, R²)
- ✅ Análisis de resultados por grupos de edad
- ✅ Visualización de predicciones continuas

---

## Tutorial 3: Comparación de Modelos CNN vs LSTM

**Objetivo**: Comparar rendimiento entre arquitecturas CNN y LSTM.

### Paso 1: Entrenar Múltiples Modelos

```bash
# Lista de modelos a comparar
declare -a models=("mobilenetv2" "resnet18" "lstm_256_2" "lstm_512_2")

# Entrenar cada modelo
for model in "${models[@]}"; do
  echo "Entrenando $model..."
  python scripts/train.py \
    --dataset common_voice \
    --task gender \
    --model $model \
    --seed 42 \
    --epochs 25 \
    --save-model \
    --experiment-name "comparison_$model"
done
```

### Paso 2: Evaluar Todos los Modelos

```bash
# Evaluar cada modelo entrenado
python scripts/evaluate.py \
  --models-dir results/ \
  --dataset common_voice \
  --task gender \
  --comparison-analysis \
  --save-plots
```

### Paso 3: Análisis Comparativo

```python
# comparar_modelos.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Resultados de evaluación
results = {
    'mobilenetv2': {'accuracy': 0.943, 'f1': 0.942, 'params': 2.2, 'inference_time': 0.012},
    'resnet18': {'accuracy': 0.938, 'f1': 0.937, 'params': 11.2, 'inference_time': 0.018},
    'lstm_256_2': {'accuracy': 0.921, 'f1': 0.920, 'params': 4.0, 'inference_time': 0.025},
    'lstm_512_2': {'accuracy': 0.934, 'f1': 0.933, 'params': 16.0, 'inference_time': 0.045}
}

df = pd.DataFrame(results).T
df.reset_index(inplace=True)
df.rename(columns={'index': 'model'}, inplace=True)

# Visualización comparativa
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy vs Parámetros
axes[0,0].scatter(df['params'], df['accuracy'], s=100)
for i, model in enumerate(df['model']):
    axes[0,0].annotate(model, (df.iloc[i]['params'], df.iloc[i]['accuracy']))
axes[0,0].set_xlabel('Parámetros (M)')
axes[0,0].set_ylabel('Accuracy')
axes[0,0].set_title('Accuracy vs Complejidad del Modelo')

# F1-Score por modelo
axes[0,1].bar(df['model'], df['f1'])
axes[0,1].set_ylabel('F1-Score')
axes[0,1].set_title('F1-Score por Modelo')
axes[0,1].tick_params(axis='x', rotation=45)

# Tiempo de inferencia
axes[1,0].bar(df['model'], df['inference_time'])
axes[1,0].set_ylabel('Tiempo (s)')
axes[1,0].set_title('Tiempo de Inferencia')
axes[1,0].tick_params(axis='x', rotation=45)

# Eficiencia (accuracy/tiempo)
df['efficiency'] = df['accuracy'] / df['inference_time']
axes[1,1].bar(df['model'], df['efficiency'])
axes[1,1].set_ylabel('Accuracy/Tiempo')
axes[1,1].set_title('Eficiencia del Modelo')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("📊 Resumen de comparación:")
print(df.to_string(index=False))
```

### ✅ Resultado del Tutorial 3

Has comparado:
- ✅ CNNs vs LSTMs en la misma tarea
- ✅ Trade-off entre accuracy y complejidad
- ✅ Eficiencia computacional
- ✅ Selección de modelo óptimo

---

## Tutorial 4: Evaluación Cross-Corpus

**Objetivo**: Evaluar generalización entre diferentes datasets.

### Paso 1: Entrenar en Dataset Fuente

```bash
# Entrenar en VoxCeleb1
python scripts/train.py \
  --dataset voxceleb1 \
  --task gender \
  --model efficientnet_b0 \
  --seed 42 \
  --epochs 25 \
  --save-model \
  --experiment-name "cross_corpus_source"
```

### Paso 2: Evaluar en Dataset Objetivo

```bash
# Evaluar modelo de VoxCeleb1 en Common Voice
python scripts/evaluate.py \
  --model-path results/cross_corpus_source/best_model.pth \
  --dataset common_voice \
  --task gender \
  --cross-corpus \
  --source-dataset voxceleb1 \
  --save-plots \
  --output-dir cross_corpus_results
```

### Paso 3: Comparar con Modelo Entrenado Nativamente

```bash
# Entrenar modelo nativo en Common Voice
python scripts/train.py \
  --dataset common_voice \
  --task gender \
  --model efficientnet_b0 \
  --seed 42 \
  --epochs 25 \
  --save-model \
  --experiment-name "native_common_voice"

# Evaluar modelo nativo
python scripts/evaluate.py \
  --model-path results/native_common_voice/best_model.pth \
  --dataset common_voice \
  --task gender \
  --save-plots \
  --output-dir native_results
```

### Paso 4: Análisis de Degradación

```python
# analisis_cross_corpus.py
import json

# Cargar resultados
with open('cross_corpus_results/metrics.json', 'r') as f:
    cross_metrics = json.load(f)

with open('native_results/metrics.json', 'r') as f:
    native_metrics = json.load(f)

# Calcular degradación
degradation = {}
for metric in ['accuracy', 'f1_score']:
    native_val = native_metrics[metric]
    cross_val = cross_metrics[metric]
    degradation[metric] = {
        'native': native_val,
        'cross_corpus': cross_val,
        'degradation': native_val - cross_val,
        'relative_degradation': (native_val - cross_val) / native_val * 100
    }

print("📉 Análisis de Degradación Cross-Corpus:")
print("=" * 50)
for metric, values in degradation.items():
    print(f"{metric.upper()}:")
    print(f"  Nativo: {values['native']:.4f}")
    print(f"  Cross-corpus: {values['cross_corpus']:.4f}")
    print(f"  Degradación: {values['degradation']:.4f} ({values['relative_degradation']:.1f}%)")
    print()
```

### Paso 5: Fine-tuning Cross-Corpus

```bash
# Fine-tuning del modelo cross-corpus
python scripts/train.py \
  --dataset common_voice \
  --task gender \
  --model efficientnet_b0 \
  --seed 42 \
  --epochs 10 \
  --learning-rate 0.0001 \
  --load-model results/cross_corpus_source/best_model.pth \
  --freeze-backbone \
  --save-model \
  --experiment-name "finetuned_cross_corpus"
```

### ✅ Resultado del Tutorial 4

Has analizado:
- ✅ Transferibilidad entre datasets
- ✅ Degradación de rendimiento cross-corpus
- ✅ Efectividad del fine-tuning
- ✅ Robustez de modelos

---

## Tutorial 5: Optimización de Hiperparámetros

**Objetivo**: Encontrar hiperparámetros óptimos usando búsqueda sistemática.

### Paso 1: Definir Espacio de Búsqueda

```python
# hyperparameter_search.py
import itertools
import json
from pathlib import Path

# Definir espacio de búsqueda
search_space = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [16, 32, 64],
    'dropout': [0.3, 0.5, 0.7],
    'weight_decay': [1e-4, 1e-5, 0]
}

# Generar todas las combinaciones
combinations = list(itertools.product(*search_space.values()))
param_names = list(search_space.keys())

print(f"🔍 Espacio de búsqueda: {len(combinations)} combinaciones")

# Guardar configuraciones
configs_dir = Path("hyperparameter_configs")
configs_dir.mkdir(exist_ok=True)

for i, combo in enumerate(combinations):
    config = dict(zip(param_names, combo))
    with open(configs_dir / f"config_{i:03d}.json", 'w') as f:
        json.dump(config, f, indent=2)

print(f"✅ Configuraciones guardadas en {configs_dir}")
```

### Paso 2: Búsqueda Grid Search

```bash
# grid_search.py
#!/bin/bash

# Configurar búsqueda grid
DATASET="common_voice"
TASK="gender"
MODEL="mobilenetv2"
EPOCHS=15

# Crear directorio de resultados
mkdir -p hyperparameter_results

# Ejecutar búsqueda
for config_file in hyperparameter_configs/config_*.json; do
    config_name=$(basename "$config_file" .json)
    echo "🔍 Probando configuración: $config_name"
    
    # Extraer parámetros del JSON
    lr=$(python -c "import json; print(json.load(open('$config_file'))['learning_rate'])")
    bs=$(python -c "import json; print(json.load(open('$config_file'))['batch_size'])")
    dropout=$(python -c "import json; print(json.load(open('$config_file'))['dropout'])")
    wd=$(python -c "import json; print(json.load(open('$config_file'))['weight_decay'])")
    
    # Entrenar modelo
    python scripts/train.py \
        --dataset $DATASET \
        --task $TASK \
        --model $MODEL \
        --epochs $EPOCHS \
        --learning-rate $lr \
        --batch-size $bs \
        --dropout $dropout \
        --weight-decay $wd \
        --seed 42 \
        --save-model \
        --experiment-name "hyperparam_$config_name" \
        --output-dir "hyperparameter_results/"
    
    echo "✅ Completado: $config_name"
done
```

### Paso 3: Analizar Resultados

```python
# analizar_hyperparametros.py
import pandas as pd
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Recopilar resultados
results = []

for result_dir in glob.glob("hyperparameter_results/hyperparam_config_*"):
    config_name = Path(result_dir).name.replace("hyperparam_", "")
    
    # Cargar configuración
    config_file = f"hyperparameter_configs/{config_name}.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Cargar métricas
    metrics_file = f"{result_dir}/metrics.json"
    if Path(metrics_file).exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        result = {**config, **metrics, 'config_name': config_name}
        results.append(result)

df = pd.DataFrame(results)

# Encontrar mejores configuraciones
best_config = df.loc[df['accuracy'].idxmax()]
print("🏆 Mejor configuración:")
print(f"  Accuracy: {best_config['accuracy']:.4f}")
print(f"  Learning Rate: {best_config['learning_rate']}")
print(f"  Batch Size: {best_config['batch_size']}")
print(f"  Dropout: {best_config['dropout']}")
print(f"  Weight Decay: {best_config['weight_decay']}")

# Visualización de hiperparámetros
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Learning rate vs accuracy
sns.boxplot(data=df, x='learning_rate', y='accuracy', ax=axes[0,0])
axes[0,0].set_title('Learning Rate vs Accuracy')

# Batch size vs accuracy
sns.boxplot(data=df, x='batch_size', y='accuracy', ax=axes[0,1])
axes[0,1].set_title('Batch Size vs Accuracy')

# Dropout vs accuracy
sns.boxplot(data=df, x='dropout', y='accuracy', ax=axes[1,0])
axes[1,0].set_title('Dropout vs Accuracy')

# Weight decay vs accuracy
sns.boxplot(data=df, x='weight_decay', y='accuracy', ax=axes[1,1])
axes[1,1].set_title('Weight Decay vs Accuracy')

plt.tight_layout()
plt.show()

# Correlación entre hiperparámetros
correlation_matrix = df[['learning_rate', 'batch_size', 'dropout', 'weight_decay', 'accuracy']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlación entre Hiperparámetros y Accuracy')
plt.show()
```

### ✅ Resultado del Tutorial 5

Has optimizado:
- ✅ Búsqueda sistemática de hiperparámetros
- ✅ Análisis de sensibilidad
- ✅ Identificación de configuración óptima
- ✅ Visualización de relaciones

---

## Tutorial 6: Pipeline Personalizado

**Objetivo**: Crear un pipeline personalizado con preprocesamiento y augmentación avanzada.

### Paso 1: Preprocesamiento Personalizado

```python
# custom_preprocessing.py
import torch
import torchaudio
import numpy as np
from src.preprocessing.audio_processing import AudioProcessor

class CustomAudioProcessor(AudioProcessor):
    """Procesador de audio personalizado con augmentación."""
    
    def __init__(self, config):
        super().__init__(config)
        self.augmentation_prob = 0.5
        
    def augment_audio(self, audio, sr):
        """Aplica augmentación de audio."""
        if np.random.random() < self.augmentation_prob:
            # Time stretching
            if np.random.random() < 0.3:
                rate = np.random.uniform(0.8, 1.2)
                audio, _ = torchaudio.sox_effects.apply_effects_tensor(
                    audio.unsqueeze(0), sr, [["tempo", str(rate)]]
                )
                audio = audio.squeeze(0)
            
            # Pitch shifting
            if np.random.random() < 0.3:
                n_steps = np.random.randint(-4, 5)
                audio, _ = torchaudio.sox_effects.apply_effects_tensor(
                    audio.unsqueeze(0), sr, [["pitch", str(n_steps * 100)]]
                )
                audio = audio.squeeze(0)
            
            # Add noise
            if np.random.random() < 0.4:
                noise_factor = np.random.uniform(0.001, 0.01)
                noise = torch.randn_like(audio) * noise_factor
                audio = audio + noise
                
        return audio
    
    def preprocess_audio(self, audio_path, augment=False):
        """Preprocesamiento con augmentación opcional."""
        audio = super().preprocess_audio(audio_path)
        
        if augment and self.training:
            audio = self.augment_audio(audio, self.config['sample_rate'])
            
        return audio
```

### Paso 2: Dataset Personalizado

```python
# custom_dataset.py
import torch
from torch.utils.data import Dataset
from src.datasets.voxceleb1 import VoxCeleb1Dataset

class AugmentedVoxCeleb1Dataset(VoxCeleb1Dataset):
    """Dataset VoxCeleb1 con augmentación personalizada."""
    
    def __init__(self, data_dir, split, task, augment=True, **kwargs):
        super().__init__(data_dir, split, task, **kwargs)
        self.augment = augment
        self.audio_processor = CustomAudioProcessor(self.config['preprocessing']['audio'])
    
    def __getitem__(self, idx):
        """Cargar muestra con augmentación."""
        audio_path, label = self.samples[idx]
        
        # Preprocesar audio con augmentación
        audio = self.audio_processor.preprocess_audio(
            audio_path, 
            augment=self.augment and self.split == 'train'
        )
        
        # Extraer características
        features = self.extract_features(audio)
        
        return features, label
```

### Paso 3: Entrenador Personalizado

```python
# custom_trainer.py
import torch
import torch.nn as nn
from src.training.trainer import SpeakerProfilingTrainer

class CustomTrainer(SpeakerProfilingTrainer):
    """Entrenador personalizado con técnicas avanzadas."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = 0.1
        self.mixup_alpha = 0.2
        
    def mixup_data(self, x, y, alpha=1.0):
        """Aplicar mixup augmentation."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Criterio de pérdida para mixup."""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def train_epoch(self, epoch):
        """Época de entrenamiento con técnicas personalizadas."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Criterio con label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Aplicar mixup
            if np.random.random() < 0.5:  # 50% de probabilidad
                data, targets_a, targets_b, lam = self.mixup_data(
                    data, targets, self.mixup_alpha
                )
                
                outputs = self.model(data)
                loss = self.mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = self.model(data)
                loss = criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Estadísticas
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
```

### Paso 4: Entrenar con Pipeline Personalizado

```python
# entrenar_personalizado.py
from torch.utils.data import DataLoader
from custom_dataset import AugmentedVoxCeleb1Dataset
from custom_trainer import CustomTrainer
from src.models.cnn_models import CNNModelFactory
from src.utils.config_utils import ConfigManager

def main():
    # Configuración
    config_manager = ConfigManager()
    config = config_manager.get_default_config()
    
    # Datasets con augmentación
    train_dataset = AugmentedVoxCeleb1Dataset(
        data_dir="data/voxceleb1",
        split="train",
        task="gender",
        augment=True,
        cache_features=False  # No cachear debido a augmentación
    )
    
    val_dataset = AugmentedVoxCeleb1Dataset(
        data_dir="data/voxceleb1",
        split="val",
        task="gender",
        augment=False
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Modelo
    model = CNNModelFactory.create_model(
        model_name="resnet18",
        num_classes=2,
        input_shape=(1, 224, 224),
        task="gender"
    )
    
    # Entrenador personalizado
    trainer = CustomTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        config=config,
        device="cuda",
        use_mixed_precision=True
    )
    
    # Entrenar
    print("🚀 Iniciando entrenamiento personalizado...")
    trainer.train_full_pipeline()
    
    print("✅ Entrenamiento completado")

if __name__ == "__main__":
    main()
```

### ✅ Resultado del Tutorial 6

Has implementado:
- ✅ Augmentación de audio personalizada
- ✅ Pipeline de entrenamiento avanzado
- ✅ Técnicas como mixup y label smoothing
- ✅ Sistema modular y extensible

---

## Tutorial 7: Reproducción de Resultados del Paper

**Objetivo**: Reproducir exactamente los resultados reportados en el paper académico.

### Paso 1: Configuración de Reproducción

```bash
# Ejecutar script de reproducción completa
python scripts/reproduce_paper.py \
  --dataset voxceleb1 \
  --task gender \
  --seeds 42,123,456,789,999,111,222,333,444,555 \
  --models all \
  --statistical-analysis \
  --sota-comparison \
  --output-dir paper_reproduction
```

### Paso 2: Verificar Configuración

```python
# verificar_configuracion.py
from src.utils.config_utils import ConfigManager

config = ConfigManager()

# Verificar configuraciones específicas del paper
paper_configs = {
    'voxceleb1': {
        'sample_rate': 16000,
        'chunk_duration': 3.0,
        'mel_bins': 224,
        'mfcc_coeffs': 40
    },
    'common_voice': {
        'sample_rate': 22050,
        'chunk_duration': 3.0,
        'mel_bins': 128,
        'mfcc_coeffs': 13
    },
    'timit': {
        'sample_rate': 16000,
        'chunk_duration': None,  # Usar longitud completa
        'mel_bins': 64,
        'mfcc_coeffs': 13
    }
}

for dataset, expected_config in paper_configs.items():
    actual_config = config.get_dataset_config(dataset)
    
    print(f"📊 Verificando {dataset}:")
    for key, expected_value in expected_config.items():
        actual_value = actual_config.get(key)
        status = "✅" if actual_value == expected_value else "❌"
        print(f"  {key}: {actual_value} (esperado: {expected_value}) {status}")
    print()
```

### Paso 3: Comparar con Resultados Reportados

```python
# comparar_resultados.py
import json
import pandas as pd

# Resultados reportados en el paper
paper_results = {
    'voxceleb1_gender': {
        'mobilenetv2': {'accuracy': 0.9564, 'f1': 0.9562},
        'efficientnet_b0': {'accuracy': 0.9621, 'f1': 0.9618},
        'resnet18': {'accuracy': 0.9587, 'f1': 0.9584},
        'resnet50': {'accuracy': 0.9603, 'f1': 0.9601},
        'vgg16': {'accuracy': 0.9542, 'f1': 0.9539},
        'alexnet': {'accuracy': 0.9498, 'f1': 0.9495},
        'densenet': {'accuracy': 0.9589, 'f1': 0.9587}
    }
}

# Cargar resultados reproducidos
with open('paper_reproduction/summary_results.json', 'r') as f:
    reproduced_results = json.load(f)

# Comparación
comparison_data = []
for dataset_task, models in paper_results.items():
    for model, paper_metrics in models.items():
        reproduced_metrics = reproduced_results[dataset_task][model]
        
        for metric in ['accuracy', 'f1']:
            paper_value = paper_metrics[metric]
            reproduced_value = reproduced_metrics[metric]['mean']
            reproduced_std = reproduced_metrics[metric]['std']
            
            difference = abs(paper_value - reproduced_value)
            within_std = difference <= reproduced_std
            
            comparison_data.append({
                'dataset_task': dataset_task,
                'model': model,
                'metric': metric,
                'paper_value': paper_value,
                'reproduced_mean': reproduced_value,
                'reproduced_std': reproduced_std,
                'difference': difference,
                'within_std': within_std
            })

df_comparison = pd.DataFrame(comparison_data)

# Análisis de reproducibilidad
reproducible_count = df_comparison['within_std'].sum()
total_count = len(df_comparison)
reproducibility_rate = reproducible_count / total_count * 100

print("📊 ANÁLISIS DE REPRODUCIBILIDAD")
print("=" * 50)
print(f"Resultados reproducibles: {reproducible_count}/{total_count} ({reproducibility_rate:.1f}%)")
print()

# Mostrar resultados por modelo
for model in df_comparison['model'].unique():
    model_data = df_comparison[df_comparison['model'] == model]
    model_reproducible = model_data['within_std'].sum()
    model_total = len(model_data)
    
    print(f"🔬 {model}: {model_reproducible}/{model_total} reproducibles")
    for _, row in model_data.iterrows():
        status = "✅" if row['within_std'] else "❌"
        print(f"  {row['metric']}: {row['reproduced_mean']:.4f}±{row['reproduced_std']:.4f} "
              f"(paper: {row['paper_value']:.4f}) {status}")
    print()
```

### Paso 4: Análisis Estadístico

```python
# analisis_estadistico.py
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Cargar resultados de múltiples seeds
results_by_seed = {}
for seed in [42, 123, 456, 789, 999, 111, 222, 333, 444, 555]:
    with open(f'paper_reproduction/seed_{seed}/results.json', 'r') as f:
        results_by_seed[seed] = json.load(f)

# Análisis de variabilidad
models = ['mobilenetv2', 'efficientnet_b0', 'resnet18']
metrics = ['accuracy', 'f1_score']

for model in models:
    print(f"📈 Análisis estadístico - {model}")
    print("-" * 40)
    
    for metric in metrics:
        values = [results_by_seed[seed]['voxceleb1_gender'][model][metric] 
                 for seed in results_by_seed.keys()]
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        ci_95 = stats.t.interval(0.95, len(values)-1, 
                                loc=mean_val, 
                                scale=stats.sem(values))
        
        print(f"  {metric}:")
        print(f"    Media: {mean_val:.4f} ± {std_val:.4f}")
        print(f"    IC 95%: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
        print(f"    Min-Max: [{min(values):.4f}, {max(values):.4f}]")
    print()

# Test de significancia vs baseline
baseline_model = 'mobilenetv2'
for model in ['efficientnet_b0', 'resnet18']:
    baseline_acc = [results_by_seed[seed]['voxceleb1_gender'][baseline_model]['accuracy'] 
                    for seed in results_by_seed.keys()]
    model_acc = [results_by_seed[seed]['voxceleb1_gender'][model]['accuracy'] 
                 for seed in results_by_seed.keys()]
    
    t_stat, p_value = stats.ttest_rel(model_acc, baseline_acc)
    significance = "Significativo" if p_value < 0.05 else "No significativo"
    
    print(f"🔬 Test t-pareado: {model} vs {baseline_model}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Resultado: {significance}")
    print()
```

### ✅ Resultado del Tutorial 7

Has reproducido:
- ✅ Resultados exactos del paper académico
- ✅ Análisis estadístico con múltiples seeds
- ✅ Tests de significancia estadística
- ✅ Validación de mejoras reportadas

---

## 🎯 Resumen de Tutoriales

¡Felicidades! Has completado todos los tutoriales del sistema de Speaker Profiling:

### ✅ Habilidades Adquiridas

1. **Entrenamiento básico** - Clasificación de género y edad
2. **Regresión de edad** - Métricas y análisis específicos
3. **Comparación de modelos** - CNNs vs LSTMs
4. **Evaluación cross-corpus** - Generalización entre datasets
5. **Optimización** - Búsqueda de hiperparámetros
6. **Personalización** - Pipelines y augmentación avanzada
7. **Reproducibilidad** - Validación científica rigurosa

### 🚀 Próximos Pasos

- Explora los [notebooks de ejemplo](../notebooks/) para análisis interactivos
- Consulta la [documentación de la API](api/README.md) para uso avanzado
- Revisa los [scripts de ejemplo](../scripts/) para automatización
- Contribuye al proyecto en [GitHub](https://github.com/tu-usuario/speaker-profiling-benchmark)

---

**¡Excelente trabajo dominando el Speaker Profiling!** 🎤✨ 