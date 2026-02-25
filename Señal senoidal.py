import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# ==================== DEFINICIONES INICIALES ====================
fs = 1000  # Frecuencia de muestreo
t = np.linspace(0, 1, fs)  # Tiempo: 0 a 1 segundo con 1000 muestras

# ==================== CREAR SEÑAL SENOIDAL ====================
f = 5  # Frecuencia de 5 Hz
senal = np.sin(2 * np.pi * f * t)

print("="*70)
print("SEÑAL SENOIDAL BÁSICA")
print("="*70)
print(f"\nParámetros:")
print(f"  • Frecuencia (f): {f} Hz")
print(f"  • Frecuencia de muestreo (fs): {fs} Hz")
print(f"  • Duración: {len(t)/fs} segundos")
print(f"  • Número de muestras: {len(t)}")
print(f"  • Amplitud: 1.0")
print(f"\nFórmula: senal = sin(2π × {f} × t)")

# ==================== GRÁFICA SIMPLE ====================
fig1, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(t, senal, linewidth=2.5, color='#1f77b4', label=f'Señal {f} Hz')
ax1.set_title("Señal Senoidal", fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel("Tiempo (s)", fontsize=13, fontweight='bold')
ax1.set_ylabel("Amplitud", fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=12, loc='upper right')
ax1.set_xlim(0, 1)
ax1.set_ylim(-1.5, 1.5)

# Agregar anotación
ax1.text(0.5, -1.3, f'Período = {1/f} seg | Ciclos = {f}', 
         ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('senal_senoidal_basica.png', dpi=150, bbox_inches='tight', facecolor='white')
print("\n✅ Gráfica 1 guardada: senal_senoidal_basica.png")
plt.show(block=False)
plt.pause(1)

# ==================== GRÁFICA MEJORADA CON DETALLES ====================
fig2 = plt.figure(figsize=(16, 10))

# Subplot 1: Señal completa
ax2 = plt.subplot(2, 2, 1)
ax2.plot(t, senal, linewidth=2.5, color='#1f77b4', label=f'Señal {f} Hz')
ax2.fill_between(t, senal, alpha=0.3, color='#1f77b4')
ax2.set_title("Señal Senoidal Completa", fontsize=13, fontweight='bold')
ax2.set_xlabel("Tiempo (s)", fontsize=11, fontweight='bold')
ax2.set_ylabel("Amplitud", fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=11)

# Subplot 2: Zoom en los primeros 0.4 segundos
ax3 = plt.subplot(2, 2, 2)
t_zoom = t[:400]
senal_zoom = senal[:400]
ax3.plot(t_zoom, senal_zoom, linewidth=3, color='#ff7f0e', marker='o', markersize=4, label='Zoom (0-0.4s)')
ax3.set_title("Zoom en los Primeros 400ms", fontsize=13, fontweight='bold')
ax3.set_xlabel("Tiempo (s)", fontsize=11, fontweight='bold')
ax3.set_ylabel("Amplitud", fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(fontsize=11)

# Marcar máximos y mínimos (CORREGIDO)
# Buscar máximos locales
maximos_idx = []
minimos_idx = []

for i in range(1, len(senal_zoom) - 1):
    if senal_zoom[i] > senal_zoom[i-1] and senal_zoom[i] > senal_zoom[i+1]:
        maximos_idx.append(i)
    elif senal_zoom[i] < senal_zoom[i-1] and senal_zoom[i] < senal_zoom[i+1]:
        minimos_idx.append(i)

if len(maximos_idx) > 0:
    ax3.plot(t_zoom[maximos_idx], senal_zoom[maximos_idx], 'g^', markersize=10, label='Máximos')
if len(minimos_idx) > 0:
    ax3.plot(t_zoom[minimos_idx], senal_zoom[minimos_idx], 'rv', markersize=10, label='Mínimos')
ax3.legend(fontsize=10)

# Subplot 3: Transformada de Fourier
ax4 = plt.subplot(2, 2, 3)
espectro = fft(senal)
magnitudes = np.abs(espectro) / len(senal)
frecuencias = fftfreq(len(senal), 1/fs)

# Solo frecuencias positivas
idx_positivas = frecuencias >= 0
ax4.plot(frecuencias[idx_positivas], magnitudes[idx_positivas], linewidth=2.5, color='#2ca02c')
ax4.fill_between(frecuencias[idx_positivas], magnitudes[idx_positivas], alpha=0.3, color='#2ca02c')
ax4.axvline(x=f, color='red', linestyle='--', linewidth=2.5, label=f'Pico en {f} Hz')
ax4.set_title("Transformada de Fourier (Espectro)", fontsize=13, fontweight='bold')
ax4.set_xlabel("Frecuencia (Hz)", fontsize=11, fontweight='bold')
ax4.set_ylabel("Magnitud", fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_xlim(0, 50)
ax4.legend(fontsize=11)

# Subplot 4: Información y estadísticas
ax5 = plt.subplot(2, 2, 4)
ax5.axis('off')

# Calcular estadísticas
amplitud_max = np.max(np.abs(senal))
amplitud_rms = np.sqrt(np.mean(senal**2))
potencia = np.mean(senal**2)
periodo = 1/f
energia = np.sum(senal**2) / fs

# Encontrar el pico en la FFT
idx_pico = np.argmax(magnitudes[:len(magnitudes)//2])
frecuencia_detectada = frecuencias[idx_positivas][idx_pico]

info_text = f"""
📊 INFORMACIÓN DE LA SEÑAL SENOIDAL
{'='*50}

⚙️  PARÁMETROS DE LA ONDA:
   • Frecuencia (f): {f} Hz
   • Período (T = 1/f): {periodo:.4f} segundos
   • Amplitud: 1.0
   • Ecuación: sin(2π × {f} × t)

📈 ESTADÍSTICAS:
   • Amplitud Máxima: {amplitud_max:.4f}
   • Amplitud RMS: {amplitud_rms:.4f}
   • Potencia Promedio: {potencia:.4f}
   • Energía Total: {energia:.4f}

🔍 ANÁLISIS DE FOURIER:
   • Frecuencia Detectada: {frecuencia_detectada:.2f} Hz
   • Magnitud del Pico: {magnitudes[idx_pico]:.4f}
   • Frecuencia Nyquist: {fs/2:.0f} Hz

⏱️  MUESTREO:
   • Frecuencia de Muestreo (fs): {fs} Hz
   • Duración: {len(t)/fs:.3f} segundos
   • Número de Muestras: {len(t)}
   • Resolución: {1/fs*1000:.2f} ms

✅ La señal fue muestreada correctamente
   porque fs > 2 × f (Teorema de Nyquist)
   {fs} > 2 × {f} = {2*f} ✓
"""

ax5.text(0.05, 0.95, info_text, transform=ax5.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.85, pad=1, 
                  edgecolor='black', linewidth=2))

plt.suptitle("Análisis Completo de Señal Senoidal 5 Hz", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('senal_senoidal_analisis.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✅ Gráfica 2 guardada: senal_senoidal_analisis.png")
plt.show(block=False)
plt.pause(1)

# ==================== GRÁFICA 3: COMPARACIÓN DE MÚLTIPLES CICLOS ====================
fig3, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1 ciclo
t1 = np.linspace(0, 1/f, 100)
senal1 = np.sin(2 * np.pi * f * t1)
axes[0, 0].plot(t1, senal1, linewidth=3, color='#1f77b4', marker='o', markersize=5)
axes[0, 0].set_title("1 Ciclo Completo", fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel("Amplitud", fontsize=11, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0, color='k', linewidth=0.5)

# 2 ciclos
t2 = np.linspace(0, 2/f, 200)
senal2 = np.sin(2 * np.pi * f * t2)
axes[0, 1].plot(t2, senal2, linewidth=3, color='#ff7f0e', marker='s', markersize=4)
axes[0, 1].set_title("2 Ciclos Completos", fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='k', linewidth=0.5)

# 3 ciclos
t3 = np.linspace(0, 3/f, 300)
senal3 = np.sin(2 * np.pi * f * t3)
axes[1, 0].plot(t3, senal3, linewidth=3, color='#2ca02c', marker='^', markersize=4)
axes[1, 0].set_title("3 Ciclos Completos", fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel("Tiempo (s)", fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel("Amplitud", fontsize=11, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=0, color='k', linewidth=0.5)

# 5 ciclos (completa)
axes[1, 1].plot(t, senal, linewidth=2.5, color='#d62728')
axes[1, 1].set_title("5 Ciclos Completos (Señal Original)", fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel("Tiempo (s)", fontsize=11, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0, color='k', linewidth=0.5)

plt.suptitle(f"Ciclos de la Onda Senoidal ({f} Hz)", fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('senal_ciclos.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✅ Gráfica 3 guardada: senal_ciclos.png")
plt.show(block=False)
plt.pause(1)

# ==================== RESUMEN ====================
print("\n" + "="*70)
print("✅ SCRIPT COMPLETADO EXITOSAMENTE")
print("="*70)
print("""
📁 ARCHIVOS GUARDADOS:
   1. senal_senoidal_basica.png
   2. senal_senoidal_analisis.png
   3. senal_ciclos.png

📊 GRÁFICAS GENERADAS:
   1. Señal senoidal simple con parámetros
   2. Análisis completo (Señal + Zoom + FFT + Estadísticas)
   3. Comparación de múltiples ciclos

🎯 CONCEPTOS CLAVE:
   • f = 5 Hz: La onda completa 5 ciclos en 1 segundo
   • T = 0.2 s: Tiempo para un ciclo completo
   • Amplitud = 1: Valor máximo de la onda
   • Teorema de Nyquist: fs ≥ 2f para muestreo correcto

💡 APLICACIONES:
   • Generación de ondas de audio
   • Análisis de señales periódicas
   • Procesamiento de señales sísmicas
   • Modelado de oscilaciones mecánicas
""")

plt.show()