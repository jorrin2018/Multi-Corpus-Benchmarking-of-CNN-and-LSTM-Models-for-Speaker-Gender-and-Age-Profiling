# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 17:07:24 2025

@author: jlorr
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt

"""
Eliminacion de silencio
"""

# 1. Cargar el audio
audio, sr = librosa.load(
    'D:/TESIS MAESTRIA/DATASETS/TIMMIT/archive/data/TEST/DR1/FAKS0/SA1.WAV.wav',
    sr=None
)
time = np.arange(len(audio)) / sr
T_total = time[-1]

# 2. Media y desviación
mu = np.mean(audio)
sigma = np.std(audio)

# 3. Mascara de voz (Mahalanobis ≥ umbral)
threshold = 0.07
mask = np.abs((audio - mu) / sigma) >= threshold

# 4. Concatenar solo los fragmentos de voz
compressed_signal = audio[mask]
compressed_time = np.arange(len(compressed_signal)) / sr

# 5. Dibujar subplots con eje X compartido
fig, (ax_orig, ax_comp) = plt.subplots(2, 1, sharex=True, figsize=(14, 6))

# Señal original (arriba)
ax_orig.plot(time, audio, color='steelblue')
ax_orig.set_ylabel('Amplitude')
ax_orig.set_title('Original Signal')

# Señal comprimida sin silencios (abajo)
ax_comp.plot(compressed_time, compressed_signal, color='darkorange')
ax_comp.set_xlabel('Time (s)')
ax_comp.set_ylabel('Amplitude')
ax_comp.set_title('Compressed Signal (sin silencios)')

# Forzar que ambos ejes X vayan de 0 a duración original
ax_comp.set_xlim(0, T_total)

plt.tight_layout()
plt.show()



"""
Filtrado de preenfasis
"""
# 1. Pre-énfasis
alpha = 0.97
audio_pre = np.empty_like(compressed_signal)
audio_pre[0] = compressed_signal[0]
audio_pre[1:] = compressed_signal[1:] - alpha * compressed_signal[:-1]

# 2. FFT de ambas señales
n = len(compressed_signal)
freqs = np.fft.rfftfreq(n, d=1/sr)
fft_orig = np.abs(np.fft.rfft(compressed_signal))
fft_pre  = np.abs(np.fft.rfft(audio_pre))

# 3. Graficar en frecuencia con “cuadros” (solo grid mayor formando recuadros)
plt.figure(figsize=(12, 6))

# Espectro original
ax1 = plt.subplot(2,1,1)
ax1.semilogy(freqs, fft_orig, color='steelblue')
ax1.set_title('Original Signal Spectrum')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Magnitude')
ax1.set_xlim(0, sr/2)
# Grid mayor en ambos ejes formando cuadros
ax1.grid(True, which='major', axis='both', linestyle='-', linewidth=0.5, color='gray')

# Espectro tras pre-énfasis
ax2 = plt.subplot(2,1,2)
ax2.semilogy(freqs, fft_pre, color='darkorange')
ax2.set_title(f'Pre-emphasized Spectrum (α={alpha})')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude')
ax2.set_xlim(0, sr/2)
# Grid mayor en ambos ejes formando cuadros
ax2.grid(True, which='major', axis='both', linestyle='-', linewidth=0.5, color='gray')

plt.tight_layout()
plt.show()
