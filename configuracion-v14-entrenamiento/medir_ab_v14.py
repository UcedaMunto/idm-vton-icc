#!/usr/bin/env python3
"""V14 - Medicion A/B del color de la prenda transferida.

Reproduce la metrica de
`configuracion-v13-entrenamiento/resultado_ab_20260830/ab_analisis.py` (fraccion de
pixeles azul/rojo/verde sobre los pixeles "vivos", hue medio, y SSIM/MAE entre pares)
pero con **rutas libres**: el original tenia las rutas de la V13 harcodeadas
(`/tmp/ab_out/base/...`, `/tmp/ab_out/v13/...`) y no se puede reutilizar tal cual.

Criterio de aceptacion del doc 04 (4.6): **torso azul > 50%** y sin artefactos
rojo/magenta. La region del torso es la de V13: filas 35%-80% del alto.

Uso:
  /home/uceda/miniconda3/envs/idm/bin/python medir_ab_v14.py \
      --referencia ruta/oficial_048400_0.jpg \
      --candidato  ruta/v14_cum3400_048400_0.jpg \
      --prenda     ruta/prenda_input.jpg \
      --etiqueta-referencia OFICIAL --etiqueta-candidato v14_cum3400
"""
import argparse
import math

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim
    HAVE_SSIM = True
except Exception:  # skimage no instalado: el resto de la metrica sigue valiendo
    HAVE_SSIM = False

# Umbrales identicos a ab_analisis.py (V13) para que los numeros sean comparables.
UMBRAL_VIVO_SAT = 0.25
UMBRAL_VIVO_VAL = 0.15
HUE_AZUL = (195, 275)
HUE_ROJO = (330, 20)   # circular: >=330 o <=20
HUE_VERDE = (80, 160)
TORSO_FILAS = (0.35, 0.80)


def load(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def rgb2hsv(img):
    """HSV en grados (0-360) / 0-1 / 0-1, sin dependencias extra."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    v = mx
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(mx > 0, d / np.maximum(mx, 1e-9), 0)
        h = np.where(d == 0, 0.0, 0.0)
        h = np.where((mx == r) & (d != 0), (60 * ((g - b) / np.maximum(d, 1e-9)) + 360) % 360, h)
        h = np.where((mx == g) & (d != 0), (60 * ((b - r) / np.maximum(d, 1e-9)) + 120) % 360, h)
        h = np.where((mx == b) & (d != 0), (60 * ((r - g) / np.maximum(d, 1e-9)) + 240) % 360, h)
    return h, s, v


def metricas(img, region=None):
    h, s, v = rgb2hsv(img)
    if region is not None:
        h, s, v = h[region], s[region], v[region]
    alive = (s > UMBRAL_VIVO_SAT) & (v > UMBRAL_VIVO_VAL)
    n = int(alive.sum())
    if n == 0:
        return {"alive": 0.0, "azul": 0.0, "rojo": 0.0, "verde": 0.0, "hue_medio": None, "sat_medio": 0.0}
    azul = (((h >= HUE_AZUL[0]) & (h <= HUE_AZUL[1])) & alive).sum() / n
    rojo = (((h >= HUE_ROJO[0]) | (h <= HUE_ROJO[1])) & alive).sum() / n
    verde = (((h >= HUE_VERDE[0]) & (h <= HUE_VERDE[1])) & alive).sum() / n
    hr = np.deg2rad(h[alive])
    hmean = math.degrees(math.atan2(np.sin(hr).mean(), np.cos(hr).mean())) % 360
    return {
        "alive": float(alive.mean()),
        "azul": float(azul),
        "rojo": float(rojo),
        "verde": float(verde),
        "hue_medio": round(hmean, 1),
        "sat_medio": float(s[alive].mean()),
    }


def comparar(a, b, region=None):
    if region is not None:
        a, b = a[region], b[region]
    mae = float(np.abs(a - b).mean())
    mse = float(((a - b) ** 2).mean())
    s = float(ssim(a, b, channel_axis=2, data_range=1.0)) if HAVE_SSIM else None
    return {"mae": mae, "mse": mse, "ssim": s}


def main():
    parser = argparse.ArgumentParser(description="V14: metrica A/B de color de la prenda transferida (doc 04, 4.6).")
    parser.add_argument("--referencia", required=True, help="Imagen de referencia (p.ej. la salida del modelo OFICIAL)")
    parser.add_argument("--candidato", required=True, help="Imagen del candidato (p.ej. la del checkpoint V14)")
    parser.add_argument("--prenda", default=None, help="Imagen de la prenda de entrada (opcional, se mide completa)")
    parser.add_argument("--etiqueta-referencia", default="referencia")
    parser.add_argument("--etiqueta-candidato", default="candidato")
    args = parser.parse_args()

    ref = load(args.referencia)
    cand = load(args.candidato)
    if ref.shape != cand.shape:
        raise SystemExit(
            f"ERROR: las imagenes tienen distinto tamano {ref.shape} vs {cand.shape}; "
            "genera ambas con el mismo width/height (doc 04: mismo protocolo para las dos)."
        )
    h0, h1 = int(ref.shape[0] * TORSO_FILAS[0]), int(ref.shape[0] * TORSO_FILAS[1])
    region = (slice(h0, h1), slice(None))

    filas = []
    if args.prenda:
        filas.append(("prenda(in)", load(args.prenda), None))
    filas.append((args.etiqueta_referencia, ref, region))
    filas.append((args.etiqueta_candidato, cand, region))

    print("=" * 78)
    print(f"COLOR (region del torso: filas {int(TORSO_FILAS[0]*100)}%-{int(TORSO_FILAS[1]*100)}%)")
    print("=" * 78)
    print(f"{'imagen':<22} {'vivos%':>7} {'azul%':>7} {'rojo%':>7} {'verde%':>7} {'hue_med':>8} {'sat_med':>8}")
    resultados = {}
    for nombre, img, reg in filas:
        m = metricas(img, reg)
        resultados[nombre] = m
        hue = m["hue_medio"] if m["hue_medio"] is not None else "-"
        print(f"{nombre:<22} {m['alive']*100:6.1f}% {m['azul']*100:6.1f}% {m['rojo']*100:6.1f}% "
              f"{m['verde']*100:6.1f}% {str(hue):>8} {m['sat_medio']:8.3f}")

    print()
    print("=" * 78)
    print("SIMILITUD ESTRUCTURAL (global | region torso)")
    print("=" * 78)
    g = comparar(ref, cand)
    r = comparar(ref, cand, region)
    ssim_g = f"{g['ssim']:.4f}" if g["ssim"] is not None else "n/a"
    ssim_r = f"{r['ssim']:.4f}" if r["ssim"] is not None else "n/a"
    print(f"{args.etiqueta_referencia} vs {args.etiqueta_candidato} | "
          f"MAE global={g['mae']:.4f} SSIM={ssim_g} | MAE torso={r['mae']:.4f} SSIM={ssim_r}")

    print()
    print("=" * 78)
    print("VEREDICTO (doc 04: torso azul > 50% y sin artefactos rojo/magenta)")
    print("=" * 78)
    for nombre in (args.etiqueta_referencia, args.etiqueta_candidato):
        m = resultados[nombre]
        azul, rojo = m["azul"] * 100, m["rojo"] * 100
        if azul > 50.0 and rojo < 20.0:
            estado = f"PASA (azul {azul:.1f}% > 50% y rojo {rojo:.1f}% bajo)"
        elif azul > rojo:
            estado = f"DUDA (azul {azul:.1f}% > rojo {rojo:.1f}%, pero azul <= 50%)"
        else:
            estado = f"FALLA (rojo {rojo:.1f}% >= azul {azul:.1f}%: la prenda NO se transfiere)"
        print(f"  {nombre}: {estado}")


if __name__ == "__main__":
    main()
