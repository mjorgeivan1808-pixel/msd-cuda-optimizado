# msd-cuda-optimizado
MSD y momentos hasta orden 4 en GPU con PyTorch + FFT. Procesa miles de trayectorias en segundos. Basado en DCM.f.
# MSD CUDA Optimizado (msd_cuda_optimizado.py)

Versión Python + PyTorch del cálculo de Desplazamiento Cuadrático Medio (MSD) y momentos hasta orden 4, usando **FFT en GPU** para máxima velocidad.

Inspirado en `DCM.f`, pero completamente reescrito para explotar la paralelización masiva de CUDA a través de PyTorch. Reduce el tiempo de cálculo de horas a segundos para miles de trayectorias.

## 🧠 ¿Por qué es tan rápido?

- Elimina los bucles anidados sobre retardos (`τ`) mediante correlaciones cruzadas con FFT.
- Cada trayectoria se procesa con unas pocas FFTs grandes (O(N log N)), saturando la GPU.
- Acumula los momentos en el *ensemble* directamente en GPU.

## 📦 Requisitos

- Python 3.8+
- PyTorch con soporte CUDA ([instrucciones](https://pytorch.org/get-started/locally/))
- NumPy

Instalación rápida:
```bash
pip install torch numpy
