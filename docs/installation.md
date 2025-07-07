# 🛠️ Guía de Instalación - Speaker Profiling

Guía completa para instalar y configurar el sistema de **Multi-Corpus Benchmarking para Speaker Profiling** en diferentes entornos.

## 📋 Índice

1. [Requisitos del Sistema](#requisitos-del-sistema)
2. [Instalación Básica](#instalación-básica)
3. [Instalación con GPU](#instalación-con-gpu)
4. [Instalación con Docker](#instalación-con-docker)
5. [Configuración de Datasets](#configuración-de-datasets)
6. [Verificación de la Instalación](#verificación-de-la-instalación)
7. [Solución de Problemas](#solución-de-problemas)

---

## 📦 Requisitos del Sistema

### Requisitos Mínimos

- **Python**: 3.8 o superior
- **RAM**: 8 GB mínimo, 16 GB recomendado
- **Almacenamiento**: 50 GB libres (datasets + modelos)
- **Sistema Operativo**: Linux, macOS, Windows 10/11

### Requisitos Recomendados

- **Python**: 3.9 o 3.10
- **RAM**: 32 GB o más
- **GPU**: NVIDIA con 8 GB VRAM (RTX 3070/4060 o superior)
- **CUDA**: 11.8 o 12.0
- **Almacenamiento SSD**: 100 GB libres

### Dependencias del Sistema

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git
sudo apt install -y libsndfile1 ffmpeg sox libsox-fmt-all
sudo apt install -y build-essential cmake
```

#### macOS
```bash
# Instalar Homebrew si no está disponible
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Instalar dependencias
brew install python git ffmpeg sox cmake
```

#### Windows
```powershell
# Instalar a través de winget (Windows 11) o descargar manualmente
winget install Python.Python.3.10
winget install Git.Git
winget install FFmpeg.FFmpeg

# O usar Chocolatey
choco install python git ffmpeg
```

---

## 🚀 Instalación Básica

### 1. Clonar el Repositorio

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/speaker-profiling-benchmark.git
cd speaker-profiling-benchmark

# Verificar estructura
ls -la
```

### 2. Crear Entorno Virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno (Linux/macOS)
source venv/bin/activate

# Activar entorno (Windows)
venv\Scripts\activate

# Verificar Python
python --version  # Debería mostrar 3.8+
```

### 3. Instalar Dependencias

```bash
# Actualizar pip
pip install --upgrade pip

# Instalar dependencias principales
pip install -r requirements.txt

# Instalar dependencias de desarrollo (opcional)
pip install -r requirements-dev.txt

# Instalar el paquete en modo desarrollo
pip install -e .
```

### 4. Verificar Instalación Básica

```bash
# Verificar importación
python -c "import speaker_profiling; print('✅ Instalación básica exitosa')"

# Verificar configuración
python -c "from src.utils.config_utils import ConfigManager; cm = ConfigManager(); print('✅ Configuración cargada')"
```

---

## 🔥 Instalación con GPU

### CUDA y PyTorch

#### Verificar CUDA Disponible

```bash
# Verificar driver NVIDIA
nvidia-smi

# Verificar versión CUDA
nvcc --version
```

#### Instalar PyTorch con CUDA

```bash
# CUDA 11.8 (recomendado)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verificar instalación GPU
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'Dispositivos: {torch.cuda.device_count()}')"
```

#### Configurar Variables de Entorno

```bash
# Agregar al ~/.bashrc o ~/.zshrc
export CUDA_VISIBLE_DEVICES=0  # Usar solo GPU 0
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# Reload shell
source ~/.bashrc
```

### Optimizaciones de Rendimiento

```bash
# Instalar librerías optimizadas
pip install nvidia-ml-py3
pip install apex  # Para mixed precision (opcional)

# Configurar memoria GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## 🐳 Instalación con Docker

### Docker CPU

```dockerfile
# Dockerfile.cpu
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    ffmpeg \
    sox \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "-c", "print('Speaker Profiling System Ready')"]
```

```bash
# Construir imagen
docker build -f Dockerfile.cpu -t speaker-profiling:cpu .

# Ejecutar contenedor
docker run -it --rm -v $(pwd)/data:/app/data speaker-profiling:cpu bash
```

### Docker GPU

```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:11.8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Instalar Python y dependencias
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-pip \
    git \
    libsndfile1 \
    ffmpeg \
    sox \
    libsox-fmt-all \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar PyTorch con CUDA
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copiar y instalar código
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "-c", "import torch; print(f'CUDA: {torch.cuda.is_available()}')"]
```

```bash
# Construir imagen GPU
docker build -f Dockerfile.gpu -t speaker-profiling:gpu .

# Ejecutar con GPU
docker run --gpus all -it --rm -v $(pwd)/data:/app/data speaker-profiling:gpu bash
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  speaker-profiling:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    volumes:
      - ./data:/app/data
      - ./results:/app/results
      - ./notebooks:/app/notebooks
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8888:8888"  # Jupyter
    command: jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

```bash
# Ejecutar con docker-compose
docker-compose up -d
```

---

## 📊 Configuración de Datasets

### Estructura de Directorios

```
data/
├── voxceleb1/
│   ├── wav/
│   │   ├── id10001/
│   │   │   ├── 1zcIwhmdeo4/
│   │   │   │   ├── 00001.wav
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   ├── vox1_meta.csv
│   └── splits/
│       ├── train_list.txt
│       ├── val_list.txt
│       └── test_list.txt
├── common_voice/
│   ├── clips/
│   │   ├── common_voice_en_001.mp3
│   │   └── ...
│   ├── validated.tsv
│   ├── train.tsv
│   ├── dev.tsv
│   └── test.tsv
└── timit/
    ├── train/
    │   ├── dr1/
    │   │   ├── fcjf0/
    │   │   │   ├── sa1.wav
    │   │   │   └── ...
    │   │   └── ...
    │   └── ...
    ├── test/
    └── spkrinfo.txt
```

### Descarga de Datasets

#### VoxCeleb1
```bash
# Crear directorio
mkdir -p data/voxceleb1

# Descargar (requiere registro)
# http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
wget -O data/voxceleb1/vox1_dev_wav.zip [URL_DEL_DATASET]

# Extraer
cd data/voxceleb1
unzip vox1_dev_wav.zip
```

#### Common Voice
```bash
# Crear directorio
mkdir -p data/common_voice

# Descargar Common Voice 17.0 (English)
# https://commonvoice.mozilla.org/en/datasets
wget -O data/common_voice/cv-corpus-17.0-2024-03-15-en.tar.gz [URL_DEL_DATASET]

# Extraer
cd data/common_voice
tar -xzf cv-corpus-17.0-2024-03-15-en.tar.gz
```

#### TIMIT
```bash
# TIMIT requiere licencia académica
# https://catalog.ldc.upenn.edu/LDC93S1

# Una vez obtenido:
mkdir -p data/timit
# Copiar archivos según estructura esperada
```

### Configuración Automática

```bash
# Script de configuración automática
python scripts/setup_datasets.py --download-samples

# Verificar configuración
python scripts/verify_datasets.py
```

### Configuración Manual

```python
# setup_data.py
from src.utils.data_utils import setup_dataset_structure

# Configurar VoxCeleb1
setup_dataset_structure(
    dataset_name="voxceleb1",
    data_dir="data/voxceleb1",
    create_splits=True
)

# Configurar Common Voice
setup_dataset_structure(
    dataset_name="common_voice", 
    data_dir="data/common_voice",
    create_splits=True
)
```

---

## ✅ Verificación de la Instalación

### Script de Verificación Completa

```python
# verify_installation.py
import sys
import torch
import librosa
import numpy as np
from pathlib import Path

def verify_installation():
    """Verifica que la instalación esté completa y funcional."""
    
    print("🔍 Verificando instalación...")
    
    # 1. Verificar Python
    print(f"✅ Python {sys.version}")
    
    # 2. Verificar PyTorch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name()}")
    
    # 3. Verificar dependencias de audio
    try:
        import librosa
        print(f"✅ Librosa {librosa.__version__}")
    except ImportError:
        print("❌ Librosa no encontrada")
        return False
    
    # 4. Verificar módulos del sistema
    try:
        from src.utils.config_utils import ConfigManager
        from src.models.cnn_models import CNNModelFactory
        print("✅ Módulos del sistema")
    except ImportError as e:
        print(f"❌ Error importando módulos: {e}")
        return False
    
    # 5. Verificar configuración
    try:
        config_manager = ConfigManager()
        config = config_manager.get_default_config()
        print("✅ Configuración cargada")
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        return False
    
    # 6. Verificar estructura de datos
    data_dir = Path("data")
    if data_dir.exists():
        datasets = [d.name for d in data_dir.iterdir() if d.is_dir()]
        print(f"✅ Datasets encontrados: {datasets}")
    else:
        print("⚠️ Directorio data no encontrado")
    
    # 7. Test básico de funcionalidad
    try:
        # Test modelo
        model = CNNModelFactory.create_model(
            model_name="mobilenetv2",
            num_classes=2,
            input_shape=(1, 224, 224),
            task="gender"
        )
        
        # Test forward pass
        dummy_input = torch.randn(1, 1, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        print("✅ Test de modelo exitoso")
        
    except Exception as e:
        print(f"❌ Error en test de modelo: {e}")
        return False
    
    print("\n🎉 ¡Instalación verificada exitosamente!")
    return True

if __name__ == "__main__":
    verify_installation()
```

```bash
# Ejecutar verificación
python verify_installation.py
```

### Tests Unitarios

```bash
# Ejecutar tests
python -m pytest tests/ -v

# Tests con cobertura
python -m pytest tests/ --cov=src --cov-report=html

# Tests específicos
python -m pytest tests/test_models.py -v
```

### Benchmark de Rendimiento

```bash
# Test de rendimiento
python scripts/benchmark_performance.py --device cuda --batch-size 32
```

---

## 🔧 Solución de Problemas

### Problemas Comunes

#### 1. Error de Importación de Módulos

**Error:**
```
ModuleNotFoundError: No module named 'src'
```

**Solución:**
```bash
# Verificar instalación en modo desarrollo
pip install -e .

# O agregar al PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 2. Error CUDA/GPU

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solución:**
```python
# Reducir batch size
config["training"]["batch_size"] = 16

# Limpiar cache GPU
torch.cuda.empty_cache()

# Usar gradient checkpointing
trainer.enable_gradient_checkpointing()
```

#### 3. Error de Audio

**Error:**
```
LibsndfileError: Error opening file
```

**Solución:**
```bash
# Instalar dependencias de audio
sudo apt install libsndfile1-dev ffmpeg

# O usar conda
conda install -c conda-forge libsndfile ffmpeg
```

#### 4. Error de Permisos

**Error:**
```
PermissionError: [Errno 13] Permission denied
```

**Solución:**
```bash
# Cambiar permisos
chmod +x scripts/*.py

# O ejecutar con python explícitamente
python scripts/train.py
```

### Diagnóstico Avanzado

```python
# diagnose.py
import torch
import subprocess
import pkg_resources

def system_diagnosis():
    """Diagnóstico completo del sistema."""
    
    print("🔍 DIAGNÓSTICO DEL SISTEMA")
    print("=" * 50)
    
    # Información del sistema
    print("Sistema:")
    print(f"  Platform: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  PyTorch: {torch.__version__}")
    
    # Memoria
    if torch.cuda.is_available():
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  GPU Used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    
    # Paquetes instalados
    installed_packages = [d.project_name for d in pkg_resources.working_set]
    print(f"  Paquetes instalados: {len(installed_packages)}")
    
    # Test de velocidad
    if torch.cuda.is_available():
        device = torch.device("cuda")
        x = torch.randn(1000, 1000).to(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        y = torch.mm(x, x)
        end.record()
        
        torch.cuda.synchronize()
        print(f"  GPU Speed Test: {start.elapsed_time(end):.2f} ms")

if __name__ == "__main__":
    system_diagnosis()
```

### Logs de Debug

```bash
# Habilitar logging detallado
export SPEAKER_PROFILING_LOG_LEVEL=DEBUG

# Ver logs en tiempo real
tail -f results/*/train.log

# Buscar errores específicos
grep -r "ERROR" results/*/
```

---

## 🚀 Configuración Avanzada

### Variables de Entorno

```bash
# .env
SPEAKER_PROFILING_DATA_DIR=/path/to/data
SPEAKER_PROFILING_RESULTS_DIR=/path/to/results
SPEAKER_PROFILING_CACHE_DIR=/tmp/speaker_profiling_cache
SPEAKER_PROFILING_LOG_LEVEL=INFO
CUDA_VISIBLE_DEVICES=0,1
TORCH_HOME=/path/to/torch/cache
```

### Configuración de Cluster

```bash
# Para entornos multi-GPU
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

# SLURM (si disponible)
sbatch --gres=gpu:2 --mem=32G scripts/slurm_train.sh
```

### Optimizaciones de Rendimiento

```python
# Configuraciones optimizadas
torch.backends.cudnn.benchmark = True  # Para tamaños de input fijos
torch.set_float32_matmul_precision('high')  # Para A100/H100
```

---

## 📞 Obtener Ayuda

### Recursos de Soporte

1. **Documentación**: [docs/api/README.md](api/README.md)
2. **Ejemplos**: [notebooks/](../notebooks/)
3. **Issues**: [GitHub Issues](https://github.com/tu-usuario/speaker-profiling-benchmark/issues)
4. **Discusiones**: [GitHub Discussions](https://github.com/tu-usuario/speaker-profiling-benchmark/discussions)

### Reportar Problemas

Al reportar un problema, incluye:

1. **Información del sistema**:
   ```bash
   python verify_installation.py > system_info.txt
   ```

2. **Logs de error**:
   ```bash
   # Últimas líneas del log
   tail -50 results/*/train.log
   ```

3. **Comando ejecutado**:
   ```bash
   python scripts/train.py --dataset voxceleb1 --task gender --model mobilenetv2
   ```

4. **Configuración utilizada**:
   ```bash
   cat config/datasets.yaml
   ```

---

**¡Instalación completada!** 🎉 

Continúa con la [Guía de Inicio Rápido](quick_start.md) para comenzar a usar el sistema. 