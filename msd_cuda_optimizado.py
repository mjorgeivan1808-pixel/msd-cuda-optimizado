# -*- coding: utf-8 -*-
"""
msd_cuda_optimizado.py
======================
Version Python + CUDA (PyTorch) de DCM.f / msd_cuda.py.

Misma fisica que msd_optimizado.py, pero las FFT y las reducciones
corren en GPU. La matematica (sumas por desfase via prefix-sums + FFT)
es identica a la version NumPy, que ya esta validada contra el doble
bucle exacto de DCM.f.

Por que es MUCHO mas rapido que tu msd_cuda.py original
-------------------------------------------------------
Tu version lanzaba ~9 reducciones x 3000 desfases = ~27000 kernels
diminutos; el overhead de lanzamiento domina y la GPU queda ociosa.
Aqui cada trayectoria se procesa con ~10 FFT grandes (O(N log N)),
saturando la GPU. El bucle de 3000 'tau' desaparece por completo.

Requisitos:  pip install torch   (con build CUDA para usar GPU)
Uso:
    python msd_cuda_optimizado.py archivo1.dat [archivo2.dat ...]
    python msd_cuda_optimizado.py "trayectorias/t*.dat"
    python msd_cuda_optimizado.py --selftest
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import glob
import math
import argparse
import numpy as np
import torch


# ============================================================
# CONFIG  (igual que la version NumPy)
# ============================================================

CONFIG = dict(
    x_col=1,
    y_col=2,
    comments="#",
    sigma=1.0,            # = size1 * Rg / 10.0 para reproducir DCM.f
    fps=1.0,
    max_lag=3000,
    vanhove_lags=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 60, 80, 100),
    vanhove_bins=201,
    dtype=torch.float64,  # float64 = preciso (recomendado para momentos 3/4).
                          # Cambia a torch.float32 si quieres mas velocidad.
    out_msd="resultado_msd_cuda.csv",
    out_vanhove_prefix="vanhove_lag",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Correlacion por FFT en GPU:  C[tau] = sum_t u[t+tau]*v[t]
# ============================================================

def _next_pow2(m):
    return 1 << int(math.ceil(math.log2(m)))


def _corr(u, v, L):
    N = u.shape[0]
    n = _next_pow2(2 * N)
    U = torch.fft.rfft(u, n)
    V = torch.fft.rfft(v, n)
    c = torch.fft.irfft(U * torch.conj(V), n)[:L + 1]
    return c


def _pref(a):
    """Suma acumulada con cero inicial: P[k] = sum(a[:k]), longitud N+1."""
    z = torch.zeros(1, dtype=a.dtype, device=a.device)
    return torch.cat((z, torch.cumsum(a, dim=0)))


def _axis_sums(x, L):
    """Sp[tau] = sum_t (x[t+tau]-x[t])^p  para p=1..4, tau=0..L  (en GPU)."""
    N = x.shape[0]
    L = min(L, N - 1)
    tau = torch.arange(L + 1, device=x.device)

    x1 = x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2

    P1, P2, P3, P4 = _pref(x1), _pref(x2), _pref(x3), _pref(x4)
    idxL = N - tau                    # indices para la suma "izquierda"
    Lp1, Rp1 = P1[idxL], P1[N] - P1[tau]
    Lp2, Rp2 = P2[idxL], P2[N] - P2[tau]
    Lp3, Rp3 = P3[idxL], P3[N] - P3[tau]
    Lp4, Rp4 = P4[idxL], P4[N] - P4[tau]

    C11 = _corr(x1, x1, L)
    C21 = _corr(x2, x1, L)
    C12 = _corr(x1, x2, L)
    C31 = _corr(x3, x1, L)
    C22 = _corr(x2, x2, L)
    C13 = _corr(x1, x3, L)

    S1 = Rp1 - Lp1
    S2 = Rp2 - 2 * C11 + Lp2
    S3 = Rp3 - 3 * C21 + 3 * C12 - Lp3
    S4 = Rp4 - 4 * C31 + 6 * C22 - 4 * C13 + Lp4
    return S1, S2, S3, S4


# ============================================================
# Acumulador de ensemble (tensores en GPU)
# ============================================================

class Ensemble:
    def __init__(self, max_lag, dtype, device):
        self.L = max_lag
        self.dtype = dtype
        self.device = device
        z = lambda: torch.zeros(max_lag + 1, dtype=dtype, device=device)
        self.sdx, self.sdx2, self.sdx3, self.sdx4 = z(), z(), z(), z()
        self.sdy, self.sdy2, self.sdy3, self.sdy4 = z(), z(), z(), z()
        self.cnt = z()
        self.n_tray = 0

    def add(self, x, y):
        N = x.shape[0]
        if N < 2:
            return
        L = min(self.L, N - 1)
        sl = slice(0, L + 1)
        d1x, d2x, d3x, d4x = _axis_sums(x, L)
        d1y, d2y, d3y, d4y = _axis_sums(y, L)
        self.sdx[sl] += d1x; self.sdx2[sl] += d2x; self.sdx3[sl] += d3x; self.sdx4[sl] += d4x
        self.sdy[sl] += d1y; self.sdy2[sl] += d2y; self.sdy3[sl] += d3y; self.sdy4[sl] += d4y
        tau = torch.arange(L + 1, device=self.device, dtype=self.dtype)
        self.cnt[sl] += (N - tau)
        self.n_tray += 1

    def results(self, fps):
        tau = torch.arange(self.L + 1, device=self.device, dtype=self.dtype)
        cnt = self.cnt.clone()
        cnt[cnt == 0] = float("nan")
        out = dict(
            lag=tau,
            lag_time=tau / fps,
            n_pairs=self.cnt,
            msd=(self.sdx2 + self.sdy2) / cnt,
            msd_x=self.sdx2 / cnt,
            msd_y=self.sdy2 / cnt,
            m1x=self.sdx / cnt, m1y=self.sdy / cnt,
            m2x=self.sdx2 / cnt, m2y=self.sdy2 / cnt,
            m3x=self.sdx3 / cnt, m3y=self.sdy3 / cnt,
            m4x=self.sdx4 / cnt, m4y=self.sdy4 / cnt,
            dcm_4=(self.sdx2 + self.sdy2) / (4.0 * cnt),
            dcmx_2=self.sdx2 / (2.0 * cnt),
            dcmy_2=self.sdy2 / (2.0 * cnt),
        )
        return {k: v.detach().cpu().numpy() for k, v in out.items()}


# ============================================================
# Lectura de trayectorias
# ============================================================

def cargar_trayectoria(path, cfg):
    data = np.genfromtxt(path, comments=cfg["comments"], dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    x = data[:, cfg["x_col"]] / cfg["sigma"]
    y = data[:, cfg["y_col"]] / cfg["sigma"]
    ok = np.isfinite(x) & np.isfinite(y)
    x = torch.tensor(x[ok], dtype=cfg["dtype"], device=device)
    y = torch.tensor(y[ok], dtype=cfg["dtype"], device=device)
    return x, y


def expandir_archivos(patrones):
    archivos = []
    for p in patrones:
        m = sorted(glob.glob(p))
        archivos.extend(m if m else [p])
    return archivos


# ============================================================
# van Hove (se acumulan los desplazamientos en GPU y se histograma en CPU)
# ============================================================

def van_hove(trayectorias, lags, nbins):
    salida = {}
    for tau in lags:
        dxs, dys = [], []
        for x, y in trayectorias:
            if x.shape[0] > tau:
                dxs.append((x[tau:] - x[:-tau]))
                dys.append((y[tau:] - y[:-tau]))
        if not dxs:
            continue
        dx = torch.cat(dxs).detach().cpu().numpy()
        dy = torch.cat(dys).detach().cpu().numpy()
        rango = max(np.abs(dx).max(), np.abs(dy).max())
        bins = np.linspace(-rango, rango, nbins)
        gx, edges = np.histogram(dx, bins=bins, density=True)
        gy, _ = np.histogram(dy, bins=bins, density=True)
        centros = 0.5 * (edges[:-1] + edges[1:])
        salida[tau] = (centros, gx, gy)
    return salida


# ============================================================
# MAIN
# ============================================================

def procesar(archivos, cfg):
    print("====================================")
    print("DEVICE:", device)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
    print("====================================")
    print(f"Procesando {len(archivos)} archivo(s)...")

    ens = Ensemble(cfg["max_lag"], cfg["dtype"], device)
    trayectorias = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()

    for i, a in enumerate(archivos, 1):
        try:
            x, y = cargar_trayectoria(a, cfg)
        except Exception as e:
            print(f"  [omitido] {a}: {e}")
            continue
        if x.shape[0] < 2:
            continue
        ens.add(x, y)
        trayectorias.append((x, y))
        if i % 50 == 0 or i == len(archivos):
            print(f"  {i}/{len(archivos)}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"\nTIEMPO GPU: {time.perf_counter() - t0:.4f} s")
    print(f"Trayectorias usadas: {ens.n_tray}")

    res = ens.results(cfg["fps"])

    cols = ["lag", "lag_time", "n_pairs", "msd", "msd_x", "msd_y",
            "m1x", "m1y", "m2x", "m2y", "m3x", "m3y", "m4x", "m4y",
            "dcm_4", "dcmx_2", "dcmy_2"]
    M = np.column_stack([res[c] for c in cols])[1:]
    np.savetxt(cfg["out_msd"], M, delimiter=",", header=",".join(cols), comments="")
    print(f"Guardado: {cfg['out_msd']}")

    vh = van_hove(trayectorias, cfg["vanhove_lags"], cfg["vanhove_bins"])
    for tau, (centros, gx, gy) in vh.items():
        np.savetxt(f"{cfg['out_vanhove_prefix']}{tau}.csv",
                   np.column_stack([centros, gx, gy]),
                   delimiter=",", header="bin_center,Gs_x,Gs_y", comments="")
    print(f"Guardadas {len(vh)} curvas de van Hove")

    print("\nPrimeros 10 MSD:")
    for k in range(1, 11):
        print(f"  lag {k:3d}   t={res['lag_time'][k]:.4f}   MSD={res['msd'][k]:.6g}")

    if device.type == "cuda":
        print(f"\nMemoria GPU usada: "
              f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    return res


# ============================================================
# Self-test
# ============================================================

def selftest():
    print("DEVICE:", device)
    rng = np.random.default_rng(0)
    N, L = 5000, 400
    x = np.cumsum(rng.standard_normal(N)) + 3.0
    y = np.cumsum(rng.standard_normal(N)) - 1.5

    # referencia exacta (doble bucle, CPU)
    def ref(a, L):
        out = {p: np.zeros(L + 1) for p in (1, 2, 3, 4)}
        for tau in range(1, L + 1):
            d = a[tau:] - a[:-tau]
            for p in (1, 2, 3, 4):
                out[p][tau] = (d**p).sum()
        return out
    rx = ref(x, L)

    xt = torch.tensor(x, dtype=torch.float64, device=device)
    s1, s2, s3, s4 = _axis_sums(xt, L)
    fft = {1: s1, 2: s2, 3: s3, 4: s4}
    print("Comparacion FFT(GPU) vs doble-bucle exacto:")
    ok = True
    for p in (1, 2, 3, 4):
        v = fft[p].detach().cpu().numpy()
        err = np.max(np.abs(v[1:] - rx[p][1:]) / (np.abs(rx[p][1:]) + 1e-9))
        print(f"  Dx^{p}: error relativo max = {err:.2e}")
        ok = ok and err < 1e-5
    print("RESULTADO:", "OK ✓" if ok else "FALLO ✗")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MSD + momentos + van Hove (CUDA)")
    ap.add_argument("archivos", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest or not args.archivos:
        if not args.archivos:
            print("Sin archivos: ejecutando self-test.\n")
        selftest()
        sys.exit(0)

    procesar(expandir_archivos(args.archivos), CONFIG)
