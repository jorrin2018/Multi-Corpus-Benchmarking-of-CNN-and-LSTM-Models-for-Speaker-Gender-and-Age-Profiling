# Multi‑Corpus Benchmarking of CNN & LSTM Models for Speaker Profiling  
### _Cheat Sheet de configuración y pipeline_  

> **Artículo**: “Multi‑Corpus Benchmarking of CNN and LSTM Models for Speaker Gender and Age Profiling on VoxCeleb1, Common Voice 17.0, and TIMIT”  
> **Autores**: Jorge Jorrin‑Coz et al., 2025  

---

## 1 · Datasets  

| Corpus | Parlantes (F/M) | Grabación | fs original | Etiquetas | Notas |
|--------|----------------|-----------|-------------|-----------|-------|
| VoxCeleb1 | 563 / 688 | “in‐the‑wild” | 16 kHz | género | Alta variabilidad y ruido |
| Common Voice 17.0 | 2 953 / 10 107 | crowdsourced | 44.1 kHz (↓ 22.05 kHz) | género + edad (6 rangos) | 60+ idiomas; solo inglés usado |
| TIMIT | 192 / 438 | estudio | 16 kHz | género + edad (valor real) | 8 dialectos US |

---

## 2 · Preprocesado de audio  

1. **Eliminación de silencios**  
   - Umbral adaptivo ![q](https://render.githubusercontent.com/render/math?math=q)=0.075 (rango óptimo 0.05–0.10).  
2. **Pre‑énfasis**  
   - Filtro $y[t]=x[t]-0.97\,x[t-1]$ para compensar –6 dB/oct.  
3. **Filtro paso‑bajo Butterworth**  
   - Orden 10, *fc* = 4 kHz.  
4. **Normalización de energía**  
   - Z‑score: $(x-\mu)/\sigma$ por archivo.  
5. **Re‐muestreo**  
   - VoxCeleb1/TIMIT → 16 kHz, CV → 22.05 kHz.

---

## 3 · Extracción de características  

| Feature | Parámetros | Uso |
|---------|------------|-----|
| **Espectrograma lineal** | STFT 25 ms / 10 ms, `n_fft=512`, power = 1 | Comparación |
| **Mel‑Spectrograma** | 128, 224 o 64 bins (tabla ↓) + log | Entrada a CNN |
| **MFCC** | 13 (CV y TIMIT) ó 40 coef. (VoxCeleb1) | Entrada a LSTM |

#### Coeficientes Mel por corpus  

| Corpus | `n_mels` | `n_mfcc` |
|--------|----------|----------|
| VoxCeleb1 | **224** | 40 |
| Common Voice | **128** | 13 |
| TIMIT | **64** | 13 |

---

## 4 · Modelos  

### CNN (Transfer Learning, ImageNet)  
- **Mobilenet‑V2**, **EfficientNet‑B0**, **ResNet50**, **ResNet18**, **VGG16**, **AlexNet**, **DenseNet**  
- `conv1` ajustado a 1 canal.  
- Se congela el *backbone* hasta el penúltimo bloque (ver Apéndice A).  
- Capa final ≡ 2 clases, 6 clases o regresión 1‑D.  

### LSTM  
```python
LSTM(input_size=n_mfcc,
     hidden_size=[128|256|512],
     num_layers=[1|2|3],
     dropout=0.3, bidirectional=True)
→ AvgPool → FC
```  

---

## 5 · Hiper‑parámetros globales  

| Parámetro | Valor |
|-----------|-------|
| Optimizador | **Adam** (lr 1e‑3, betas 0.9/0.999) |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 3) |
| Weight decay | 1e‑4 |
| Dropout | 0.5 |
| Batch | 64 |
| Épocas máx | 100 |
| Early‑Stopping | patience = 15 |
| Seeds | 10 (1, 12, 42, 77, 101, 128, 222, 314, 512, 999) |

---

## 6 · Flujo de entrenamiento  

1. **Stage 1 · Selección de modelo**  
   - Subset 1 (5 k/500) de VoxCeleb1.  
   - Se prueban 7 CNN + 9 configs LSTM.  
   - Métrica: *accuracy* (clas.) o MAE (reg.).  
2. **Stage 2 · Fine‑Tuning**  
   - Se entrenan los 3 CNN y 1 LSTM top en cada corpus completo.  
   - Congelar capas bajas; ajustar capas altas.  

---

## 7 · Comandos clave  

```bash
# Entorno
conda create -n spk_ablation python=3.11
conda activate spk_ablation
pip install torch torchaudio torchvision lightning librosa pingouin pandas matplotlib tqdm

# Lanzar Stage 1
python train.py --stage select --corpus VoxCeleb1 --subset 1

# Fine‑tune seleccionado
python train.py --stage finetune --corpus CommonVoice --arch ResNet50
```

---

## 8 · Pseudocódigo simplificado  

```python
for corpus in DATASETS:
    ds_train, ds_test = split(corpus, 0.8)
    best_cnns, best_lstm = stage1_select(ds_train)
    final_models = stage2_finetune(best_cnns + [best_lstm], ds_train)
    evaluate(final_models, ds_test)
```

---

## 9 · Principales resultados  

| Corpus | Métrica | SOTA previo | Propuesto | Δ |
|--------|---------|-------------|-----------|---|
| VoxCeleb1 | Acc. género | 98.29 % | **98.86 %** | +0.57 |
| Common Voice | Acc. género | 98.57 % | **99.82 %** | +1.25 |
| Common Voice | Acc. edad | 97.00 % | **99.86 %** | +2.86 |
| TIMIT | MAE edad (♂) | 5.12 | **5.35** | −0.23 |

*Todas las mejoras son significativas (paired t‑test, p < 0.01, n = 10).*  

---

## 10 · Hardware mínimo recomendado  

- **GPU** NVIDIA RTX 3070 Ti (8 GB)  
- **CPU** Ryzen 7 3700X, 32 GB RAM  
- Entrenamiento completo ≈ 8 h por corpus.

---

## 11 · Licencia y repositorio  

Código y pesos pre‑entrenados disponibles bajo **MIT License**:  
<https://github.com/tu‑usuario/speaker‑profiling‑pipeline>

---

> **Contacto**: jljorrincoz@gmail.com &nbsp;·  Última actualización: 2025‑07‑07
