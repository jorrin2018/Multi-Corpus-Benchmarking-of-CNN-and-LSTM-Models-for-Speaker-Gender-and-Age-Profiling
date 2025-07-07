# 📚 Documentación de la API - Speaker Profiling

Documentación completa de la API del sistema de **Multi-Corpus Benchmarking para Speaker Profiling**.

## 📋 Índice General

### 🔧 Módulos de Preprocesamiento
- [`audio_processing`](preprocessing/audio_processing.md) - Procesamiento de señales de audio
- [`feature_extraction`](preprocessing/feature_extraction.md) - Extracción de características

### 🏗️ Módulos de Modelos
- [`cnn_models`](models/cnn_models.md) - Factorías y arquitecturas CNN
- [`lstm_models`](models/lstm_models.md) - Factorías y arquitecturas LSTM

### 📊 Módulos de Datasets
- [`voxceleb1`](datasets/voxceleb1.md) - Dataset VoxCeleb1
- [`common_voice`](datasets/common_voice.md) - Dataset Common Voice
- [`timit`](datasets/timit.md) - Dataset TIMIT

### 🎯 Módulos de Entrenamiento
- [`trainer`](training/trainer.md) - Entrenador principal
- [`callbacks`](training/callbacks.md) - Sistema de callbacks

### 📈 Módulos de Evaluación
- [`metrics`](evaluation/metrics.md) - Calculadora de métricas
- [`benchmarking`](evaluation/benchmarking.md) - Sistema de benchmarking

### 🛠️ Módulos de Utilidades
- [`config_utils`](utils/config_utils.md) - Gestión de configuración
- [`data_utils`](utils/data_utils.md) - Utilidades de datos

---

## 🚀 Inicio Rápido

### Importaciones Básicas

```python
# Configuración
from src.utils.config_utils import ConfigManager

# Preprocesamiento
from src.preprocessing.audio_processing import AudioProcessor
from src.preprocessing.feature_extraction import FeatureExtractor

# Datasets
from src.datasets.voxceleb1 import VoxCeleb1Dataset
from src.datasets.common_voice import CommonVoiceDataset
from src.datasets.timit import TIMITDataset

# Modelos
from src.models.cnn_models import CNNModelFactory
from src.models.lstm_models import LSTMModelFactory

# Entrenamiento
from src.training.trainer import SpeakerProfilingTrainer
from src.training.callbacks import CallbackManager

# Evaluación
from src.evaluation.metrics import MetricsCalculator
```

### Ejemplo de Uso Básico

```python
# Configurar sistema
config_manager = ConfigManager()
config = config_manager.get_default_config()

# Cargar dataset
dataset = VoxCeleb1Dataset(
    data_dir="data/voxceleb1",
    split="train",
    task="gender",
    cache_features=True
)

# Crear modelo
model = CNNModelFactory.create_model(
    model_name="mobilenetv2",
    num_classes=2,
    input_shape=(1, 224, 224),
    task="gender"
)

# Entrenar
trainer = SpeakerProfilingTrainer(
    model=model,
    config=config,
    device="cuda"
)

trainer.train_full_pipeline()
```

---

## 📖 Guías por Módulo

### Preprocesamiento de Audio

El módulo de preprocesamiento maneja la carga, limpieza y transformación de señales de audio:

- **Eliminación de silencios** con parámetro q=0.075
- **Filtro pre-énfasis** con coeficiente 0.97
- **Filtrado Butterworth** de 10º orden, frecuencia de corte 4kHz
- **Normalización Z-score** para estabilidad numérica

```python
audio_processor = AudioProcessor(config["preprocessing"]["audio"])
processed_audio = audio_processor.preprocess_audio(raw_audio)
```

### Extracción de Características

Sistema flexible para extraer diferentes tipos de características:

- **Espectrogramas lineales** con ventana Hann
- **Mel-spectrogramas** específicos por corpus
- **MFCC** optimizados para cada dataset

```python
feature_extractor = FeatureExtractor(config["preprocessing"]["features"])

# Mel-spectrogram para VoxCeleb1 (224x224)
mel_spec = feature_extractor.extract_mel_spectrogram(audio, dataset="voxceleb1")

# MFCC para TIMIT (13 coeficientes)
mfcc = feature_extractor.extract_mfcc(audio, dataset="timit")
```

### Factorías de Modelos

Sistema de factorías para crear modelos de manera consistente:

#### CNN Models

Soporta 7 arquitecturas CNN con transfer learning:

```python
# Crear modelo CNN
model = CNNModelFactory.create_model(
    model_name="efficientnet_b0",  # mobilenetv2, resnet18, etc.
    num_classes=2,
    input_shape=(1, 224, 224),
    task="gender"
)
```

#### LSTM Models

Soporta 9 configuraciones LSTM bidireccionales:

```python
# Crear modelo LSTM
model = LSTMModelFactory.create_model(
    model_name="lstm_256_2",  # 256 hidden, 2 layers
    num_classes=2,
    input_size=40,  # MFCC features
    task="gender"
)
```

### Sistema de Entrenamiento

Entrenador unificado con pipeline de 2 etapas:

```python
trainer = SpeakerProfilingTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    config=config,
    device="cuda",
    use_mixed_precision=True
)

# Pipeline completo: selección + fine-tuning
trainer.train_full_pipeline()
```

### Sistema de Evaluación

Calculadora de métricas completa:

```python
metrics_calculator = MetricsCalculator(task="gender")

# Calcular todas las métricas
metrics = metrics_calculator.calculate_metrics(targets, predictions)
# Retorna: accuracy, f1_score, precision, recall, auc_roc, etc.

# Análisis estadístico
analysis = metrics_calculator.statistical_analysis(results_list)
# Retorna: confidence_intervals, significance_tests, etc.
```

---

## 🔧 Configuración Avanzada

### Estructura de Configuración

El sistema usa archivos YAML para configuración:

```yaml
datasets:
  voxceleb1:
    sample_rate: 16000
    chunk_duration: 3.0
    mel_bins: 224
    
models:
  cnn:
    - mobilenetv2
    - efficientnet_b0
    - resnet18
    # ...
  
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 50
  pipeline: "full"  # selection, finetune, full
```

### Personalización de Modelos

#### Agregar Nuevo Modelo CNN

```python
class CustomCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Implementar arquitectura personalizada
        
    def forward(self, x):
        # Implementar forward pass
        return x

# Registrar en la factoría
CNNModelFactory.register_model("custom_cnn", CustomCNN)
```

#### Personalizar Pipeline de Entrenamiento

```python
# Configurar callbacks personalizados
callback_manager = CallbackManager()
callback_manager.add_callback(EarlyStoppingCallback(patience=10))
callback_manager.add_callback(LearningRateSchedulerCallback())

trainer.set_callbacks(callback_manager)
```

---

## 📊 Métricas y Evaluación

### Métricas Disponibles

#### Para Clasificación de Género
- **Accuracy**: Precisión general
- **F1-Score**: Media armónica de precision y recall
- **Precision**: Precisión por clase
- **Recall**: Sensibilidad por clase
- **AUC-ROC**: Área bajo la curva ROC
- **Confusion Matrix**: Matriz de confusión

#### Para Estimación de Edad
- **MAE**: Error absoluto medio (regresión)
- **MSE**: Error cuadrático medio
- **RMSE**: Raíz del error cuadrático medio
- **R²**: Coeficiente de determinación
- **Accuracy**: Precisión por rangos (clasificación)

### Cross-Corpus Evaluation

Sistema para evaluar generalización entre datasets:

```python
# Entrenar en VoxCeleb1, evaluar en Common Voice
source_model = "results/voxceleb1_gender_mobilenetv2/model.pth"
target_dataset = CommonVoiceDataset(split="test", task="gender")

cross_metrics = evaluate_cross_corpus(source_model, target_dataset)
```

---

## 🔍 Debugging y Diagnóstico

### Logging Avanzado

```python
import logging

# Configurar logging detallado
logging.basicConfig(level=logging.DEBUG)

# Logs específicos por módulo
logger = logging.getLogger("speaker_profiling.training")
logger.setLevel(logging.INFO)
```

### Profiling de Rendimiento

```python
# Monitorear uso de memoria GPU
torch.cuda.memory_summary()

# Profiling de tiempo de entrenamiento
with torch.profiler.profile() as prof:
    trainer.train_epoch()

print(prof.key_averages().table())
```

### Validación de Datos

```python
# Verificar integridad del dataset
dataset.validate_integrity()

# Estadísticas del dataset
stats = dataset.get_statistics()
print(f"Speakers: {stats['num_speakers']}")
print(f"Audio files: {stats['num_files']}")
```

---

## 🚀 Extensibilidad

### Agregar Nuevo Dataset

```python
class CustomDataset(BaseSpeakerDataset):
    def __init__(self, data_dir: str, split: str, task: str):
        super().__init__(data_dir, split, task)
        self.load_metadata()
    
    def load_metadata(self):
        # Implementar carga de metadatos
        pass
    
    def __getitem__(self, idx):
        # Implementar carga de muestras
        return features, label
```

### Personalizar Métricas

```python
class CustomMetricsCalculator(MetricsCalculator):
    def calculate_custom_metric(self, y_true, y_pred):
        # Implementar métrica personalizada
        return custom_score
```

---

## 📞 Soporte Técnico

### Errores Comunes

1. **CUDA out of memory**
   ```python
   # Reducir batch_size o usar gradient_checkpointing
   config["training"]["batch_size"] = 16
   trainer.enable_gradient_checkpointing()
   ```

2. **Dataset no encontrado**
   ```python
   # Verificar estructura de directorios
   dataset.validate_data_structure()
   ```

3. **Modelo no converge**
   ```python
   # Ajustar learning rate o usar scheduler
   config["training"]["learning_rate"] = 0.0001
   trainer.use_scheduler("reduce_on_plateau")
   ```

### Contacto

Para problemas técnicos o sugerencias:
- [Issues en GitHub](https://github.com/tu-usuario/speaker-profiling-benchmark/issues)
- [Documentación completa](https://speaker-profiling-docs.readthedocs.io)
- [Ejemplos adicionales](../notebooks/)

---

**Última actualización**: 2024
**Versión de la API**: 1.0.0 