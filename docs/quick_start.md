# Guía de Inicio Rápido - Speaker Profiling

¡Bienvenido al sistema de **Multi-Corpus Benchmarking para Speaker Profiling**! Esta guía te ayudará a comenzar rápidamente con el entrenamiento y evaluación de modelos para clasificación de género y edad.

## 📋 Índice

1. [Instalación Rápida](#instalación-rápida)
2. [Configuración Inicial](#configuración-inicial)
3. [Primer Entrenamiento](#primer-entrenamiento)
4. [Evaluación de Modelos](#evaluación-de-modelos)
5. [Predicciones](#predicciones)
6. [Ejemplos Avanzados](#ejemplos-avanzados)

## 🚀 Instalación Rápida

### Requisitos Previos
- Python 3.8+
- CUDA (opcional, para GPU)
- Git

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/speaker-profiling-benchmark.git
cd speaker-profiling-benchmark

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar el paquete
pip install -e .
```

## ⚙️ Configuración Inicial

### 1. Estructura de Datos

Organiza tus datos siguiendo esta estructura:

```
data/
├── voxceleb1/
│   ├── wav/
│   ├── metadata.csv
│   └── splits/
├── common_voice/
│   ├── clips/
│   ├── metadata.csv
│   └── splits/
└── timit/
    ├── wav/
    ├── metadata.csv
    └── splits/
```

### 2. Verificar Configuración

```bash
# Verificar instalación
python -c "import speaker_profiling; print('✅ Instalación exitosa')"

# Verificar configuración
python -c "from src.utils.config_utils import ConfigManager; cm = ConfigManager(); print('✅ Configuración cargada')"
```

## 🎯 Primer Entrenamiento

### Ejemplo 1: Clasificación de Género en VoxCeleb1

```bash
# Entrenar modelo CNN básico
python scripts/train.py \
  --dataset voxceleb1 \
  --task gender \
  --model mobilenetv2 \
  --seed 42 \
  --epochs 10 \
  --save-model

# Ver progreso
tail -f results/voxceleb1_gender_mobilenetv2_seed42/train.log
```

**Salida esperada:**
```
================================================================================
INICIANDO ENTRENAMIENTO DE SPEAKER PROFILING
================================================================================
Dataset: voxceleb1
Tarea: gender
Modelo: mobilenetv2
Seed: 42
Pipeline: full

Usando device: cuda
GPU disponible: NVIDIA GeForce RTX 3080
Dataset cargado - Train: 85432, Val: 12204, Test: 12205
Modelo creado: mobilenetv2
Número de parámetros: 2,223,872

Ejecutando pipeline completo de 2 etapas...
Etapa 1/2: Selección de modelo...
Epoch 1/50: train_loss=0.4523, val_loss=0.3821, val_acc=0.8234
...
✅ Entrenamiento completado
```

### Ejemplo 2: Clasificación de Edad en Common Voice

```bash
# Entrenar modelo LSTM
python scripts/train.py \
  --dataset common_voice \
  --task age \
  --model lstm_256_2 \
  --seed 42 \
  --batch-size 64 \
  --save-model
```

### Ejemplo 3: Regresión de Edad en TIMIT

```bash
# Entrenar para regresión de edad
python scripts/train.py \
  --dataset timit \
  --task age \
  --model resnet18 \
  --seed 42 \
  --pipeline full \
  --mixed-precision \
  --save-model
```

## 📊 Evaluación de Modelos

### Evaluar Modelo Individual

```bash
# Evaluar modelo entrenado
python scripts/evaluate.py \
  --model-path results/voxceleb1_gender_mobilenetv2_seed42/final_model.pth \
  --dataset voxceleb1 \
  --task gender \
  --save-plots \
  --save-predictions
```

**Salida esperada:**
```
================================================================================
RESUMEN DE EVALUACIÓN
================================================================================
Mejor modelo: mobilenetv2
Mejor accuracy: 0.9621

Métricas detalladas:
- Accuracy: 0.9621
- F1-Score: 0.9618
- Precision: 0.9625
- Recall: 0.9612
- AUC-ROC: 0.9864

✅ Resultados guardados en: evaluation_results/
```

### Evaluación Cross-Corpus

```bash
# Entrenar en VoxCeleb1 y evaluar en Common Voice
python scripts/evaluate.py \
  --model-path results/voxceleb1_gender_mobilenetv2_seed42/final_model.pth \
  --dataset common_voice \
  --task gender \
  --cross-corpus \
  --output-dir cross_corpus_results
```

### Evaluar Múltiples Modelos

```bash
# Evaluar todos los modelos en un directorio
python scripts/evaluate.py \
  --models-dir results/ \
  --dataset voxceleb1 \
  --task gender \
  --save-plots
```

## 🎤 Predicciones

### Predicción en Archivo Individual

```bash
# Predecir género en archivo de audio
python scripts/predict.py \
  --model-path results/voxceleb1_gender_mobilenetv2_seed42/final_model.pth \
  --audio-file sample_audio.wav \
  --task gender \
  --output-probs
```

**Salida esperada:**
```
================================================================================
RESULTADOS DE PREDICCIÓN
================================================================================
Archivo: sample_audio.wav | Género: Femenino (Confianza: 0.924)
```

### Predicción en Lotes

```bash
# Predecir en múltiples archivos
python scripts/predict.py \
  --model-path results/common_voice_age_lstm_256_2_seed42/final_model.pth \
  --audio-dir audio_samples/ \
  --task age \
  --output-file predictions.csv \
  --batch-size 32
```

### Predicción con Características Personalizadas

```bash
# Usar MFCC en lugar de mel-spectrogramas
python scripts/predict.py \
  --model-path results/timit_age_resnet18_seed42/final_model.pth \
  --audio-file audio.wav \
  --task age \
  --feature-type mfcc \
  --output-features
```

## 🏁 Ejemplos Avanzados

### 1. Benchmark Completo

```bash
# Ejecutar benchmark multi-modelo
python scripts/benchmark.py \
  --dataset voxceleb1 \
  --task gender \
  --models mobilenetv2,resnet18,lstm_256_2 \
  --seeds 42,123,456 \
  --parallel-jobs 2 \
  --save-plots
```

### 2. Reproducir Resultados del Paper

```bash
# Reproducir experimentos completos del paper
python scripts/reproduce_paper.py \
  --dataset voxceleb1 \
  --task gender \
  --statistical-analysis \
  --sota-comparison
```

### 3. Pipeline Personalizado

```python
# ejemplo_personalizado.py
from src.training.trainer import SpeakerProfilingTrainer
from src.datasets.voxceleb1 import VoxCeleb1Dataset
from src.models.cnn_models import CNNModelFactory
from src.utils.config_utils import ConfigManager

# Configuración personalizada
config = ConfigManager().get_default_config()
config["training"]["learning_rate"] = 0.0005
config["training"]["batch_size"] = 16

# Crear dataset
dataset = VoxCeleb1Dataset(
    data_dir="data/voxceleb1",
    split="train",
    task="gender",
    cache_features=True
)

# Crear modelo
model = CNNModelFactory.create_model(
    model_name="efficientnet_b0",
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

## 📈 Monitoreo y Debugging

### Ver Logs en Tiempo Real

```bash
# Seguir logs de entrenamiento
tail -f results/*/train.log

# Ver métricas específicas
grep "val_acc" results/*/train.log
```

### Verificar Estado del Sistema

```bash
# Comprobar uso de GPU
nvidia-smi

# Verificar espacio en disco
df -h

# Comprobar procesos activos
ps aux | grep python
```

### Debugging Común

1. **Error de memoria GPU:**
```bash
# Reducir batch size
python scripts/train.py --batch-size 16 ...
```

2. **Dataset no encontrado:**
```bash
# Verificar estructura de datos
ls -la data/voxceleb1/
```

3. **Modelo no converge:**
```bash
# Ajustar learning rate
python scripts/train.py --lr 0.0001 ...
```

## 📝 Mejores Prácticas

### 1. Nomenclatura de Experimentos
- Usa seeds consistentes para reproducibilidad
- Nombra experimentos descriptivamente
- Documenta cambios en configuración

### 2. Gestión de Recursos
- Monitorea uso de memoria GPU
- Usa mixed precision para eficiencia
- Configura early stopping

### 3. Evaluación
- Siempre evalúa en test set
- Realiza cross-corpus evaluation
- Compara con baselines establecidos

### 4. Reproducibilidad
- Fija seeds en todos los experimentos
- Documenta versiones de dependencias
- Guarda configuraciones utilizadas

## 🔗 Enlaces Útiles

- [Documentación Completa](docs/api/README.md)
- [Notebooks de Ejemplo](notebooks/)
- [Configuración Avanzada](docs/configuration.md)
- [Solución de Problemas](docs/troubleshooting.md)
- [Contribuir](CONTRIBUTING.md)

## 🆘 Soporte

Si encuentras problemas:

1. Revisa la [documentación](docs/)
2. Consulta [issues conocidos](https://github.com/tu-usuario/speaker-profiling-benchmark/issues)
3. Crea un [nuevo issue](https://github.com/tu-usuario/speaker-profiling-benchmark/issues/new)

---

¡Feliz entrenamiento! 🎉 Para casos de uso más avanzados, consulta los [notebooks de ejemplo](notebooks/) y la [documentación completa de la API](docs/api/). 