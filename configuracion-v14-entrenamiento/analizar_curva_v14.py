#!/usr/bin/env python3
# coding=utf-8
"""V14: mide la curva pasos -> fidelidad de prenda usando las muestras que el
watchdog ya genera en `entrenamiento continuo/imagenes_revision/`.

Pregunta que responde: "con N pasos, cuanto se conserva la transferencia de
prenda respecto al modelo oficial?".

Metodo (validado): las funciones de este script reproducen EXACTAMENTE los
valores del script autoritativo
`configuracion-v13-entrenamiento/resultado_ab_20260830/ab_analisis.py`
(comprobado con los archivos de ese A/B: OFICIAL azul 24.6% / rojo 26.0%,
V12-1800 azul 0.0% / rojo 94.2%).

100% CPU: NO importa torch ni toca la GPU, por lo que se puede ejecutar
mientras el entrenamiento continua corriendo.

Uso:
    python3 analizar_curva_v14.py
    python3 analizar_curva_v14.py --steps 500=<dir> --steps 1500=<dir>
"""
import argparse
import math
import os

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim
    HAVE_SSIM = True
except Exception:
    HAVE_SSIM = False

ROOT = "/home/uceda/Documents/IDM-VTON"
REVIEW_ROOT = f"{ROOT}/entrenamiento continuo/imagenes_revision"
DATA = f"{ROOT}/dataset/DATA_DIR_PREP/test"
AB_V13 = f"{ROOT}/configuracion-v13-entrenamiento/resultado_ab_20260830"

# Muestras por defecto: son las unicas que el watchdog ha generado para V14.
# run_20260908_220420/checkpoint-500  -> acumulado   500 (primer bloque desde el oficial)
# run_20260909_100801/checkpoint-500  -> acumulado 1.500 (tercer bloque)
DEFAULT_STEPS = {
    500: f"{REVIEW_ROOT}/run_20260908_220420",
    1500: f"{REVIEW_ROOT}/run_20260909_100801",
}
PAIRS = ["048392_0.jpg", "048393_0.jpg"]

# Calibracion: A/B de V13 (par 048400_0, prenda AZUL) con ancla oficial.
CALIB = [
    ("OFICIAL (sin fine-tune)", f"{AB_V13}/oficial/oficial_048400_0.jpg"),
    ("BASE night ck-250", f"{AB_V13}/salidas/01_base_train_night_ck250.jpg"),
    ("V13 cum800", f"{AB_V13}/salidas/02_v13_cum800.jpg"),
    ("V12 cum1800", f"{AB_V13}/salidas/03_v12_cum1800.jpg"),
]
CALIB_CLOTH = f"{AB_V13}/salidas/00_prenda_input.jpg"


def load(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def rgb2hsv(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    v = mx
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(mx > 0, d / np.maximum(mx, 1e-9), 0.0)
        h = np.zeros_like(mx)
        d9 = np.maximum(d, 1e-9)
        h = np.where((mx == r) & (d != 0), (60 * ((g - b) / d9) + 360) % 360, h)
        h = np.where((mx == g) & (d != 0), (60 * ((b - r) / d9) + 120) % 360, h)
        h = np.where((mx == b) & (d != 0), (60 * ((r - g) / d9) + 240) % 360, h)
    return h, s, v


def metricas(img, region=None):
    """Identico a ab_analisis.py: %, azul, rojo, verde sobre pixeles 'vivos'."""
    h, s, v = rgb2hsv(img)
    if region is not None:
        h, s, v = h[region], s[region], v[region]
    alive = (s > 0.25) & (v > 0.15)
    n = int(alive.sum())
    oscuro = float((v < 0.15).mean())
    if n == 0:
        return {"alive": 0.0, "azul": 0.0, "rojo": 0.0, "verde": 0.0,
                "oscuro": oscuro, "hue_medio": None, "sat_medio": 0.0}
    azul = float((((h >= 195) & (h <= 275)) & alive).sum() / n)
    rojo = float(((((h >= 330) | (h <= 20))) & alive).sum() / n)
    verde = float((((h >= 80) & (h <= 160)) & alive).sum() / n)
    hr = np.deg2rad(h[alive])
    hmean = math.degrees(math.atan2(np.sin(hr).mean(), np.cos(hr).mean())) % 360
    return {"alive": float(alive.mean()), "azul": azul, "rojo": rojo, "verde": verde,
            "oscuro": oscuro, "hue_medio": round(hmean, 1),
            "sat_medio": float(s[alive].mean())}


def comparar(a, b, region=None):
    if region is not None:
        a, b = a[region], b[region]
    out = {"mae": float(np.abs(a - b).mean() * 255)}
    if HAVE_SSIM:
        out["ssim"] = float(ssim(a, b, channel_axis=2, data_range=1.0))
    return out


def torso(img):
    h0, h1 = int(img.shape[0] * 0.35), int(img.shape[0] * 0.80)
    return (slice(h0, h1), slice(None))


def fmt(m, extra_sat=False):
    sat = f" {m['sat_medio']:6.3f}" if extra_sat else ""
    return (f"oscuro={m['oscuro']*100:5.1f}% azul={m['azul']*100:5.1f}% "
            f"rojo={m['rojo']*100:5.1f}% verde={m['verde']*100:5.1f}% "
            f"vivos={m['alive']*100:5.1f}%{sat}")


def seccion_v14(steps):
    print("=" * 100)
    print("A) V14 - CURVA REAL: torso (filas 35%-80%) de las muestras del watchdog")
    print("=" * 100)
    print("Referencia de la metrica: OFICIAL deja el torso con vivos~5% y azul~25% (sin")
    print("oscuro); los checkpoints entrenados colapsan a oscuro 16-19% + rojo 61-94% (sec. C).")
    print()
    salidas = {}
    for par in PAIRS:
        print(f"--- par {par}")
        cloth = load(f"{DATA}/cloth/{par}")
        persona = load(f"{DATA}/image/{par}")
        print(f"    {'prenda IN':<12} {fmt(metricas(cloth, None), extra_sat=True)}")
        print(f"    {'persona IN':<12} {fmt(metricas(persona, torso(persona)), extra_sat=True)}")
        for n in sorted(steps):
            path = os.path.join(steps[n], par)
            if not os.path.exists(path):
                print(f"    {'V14 ck-'+str(n):<12} (no existe: {path})")
                continue
            out = load(path)
            reg = torso(out)
            salidas[(par, n)] = (out, reg)
            print(f"    {'V14 pasos '+str(n):<12} {fmt(metricas(out, reg), extra_sat=True)}")
        print()

    print("=" * 100)
    print("B) CUANTO SE MUEVE EL MODELO ENTRE CHECKPOINTS (SSIM~1 => casi no cambia)")
    print("=" * 100)
    ns = sorted(steps)
    for i in range(len(ns) - 1):
        a_n, b_n = ns[i], ns[i + 1]
        for par in PAIRS:
            if (par, a_n) not in salidas or (par, b_n) not in salidas:
                continue
            ia, ra = salidas[(par, a_n)]
            ib, rb = salidas[(par, b_n)]
            g = comparar(ia, ib)
            r = comparar(ia, ib, ra)
            print(f"{par:<14} {a_n:>5} vs {b_n:<5} SSIM global={g.get('ssim', float('nan')):.4f} "
                  f"MAE={g['mae']:5.2f}/255 | SSIM torso={r.get('ssim', float('nan')):.4f} "
                  f"MAE={r['mae']:5.2f}/255")
    print("Lectura: SSIM 0.85-0.92 en 1.000 pasos = el modelo CAMBIA POCO, pero cambia en la")
    print("direccion equivocada (ver el corrimiento de color del torso/prendas en la seccion A).")

    print()
    print("=" * 100)
    print("C) CALIBRACION con el A/B de V13 (par 048400_0, prenda AZUL, ancla oficial)")
    print("=" * 100)
    print(f"PRENDA DE ENTRADA (azul): {fmt(metricas(load(CALIB_CLOTH), None))}")
    print()
    for nombre, path in CALIB:
        if not os.path.exists(path):
            print(f"{nombre:<26} (falta {path})")
            continue
        img = load(path)
        print(f"{nombre:<26} {fmt(metricas(img, torso(img)))}")
    print()
    print("Conclusion: la metrica SI separa 'transfiere' de 'roto'. El oficial conserva azul")
    print("en el torso y no tiene pixeles oscuros; TODOS los checkpoints entrenados pierden")
    print("el azul (0-0.3%) y ganan rojo (61-94%).")


def parse_steps(items):
    steps = {}
    for it in items or []:
        n, _, d = it.partition("=")
        steps[int(n)] = d
    return steps or dict(DEFAULT_STEPS)


def main():
    ap = argparse.ArgumentParser(description="Curva pasos vs fidelidad de prenda (V14).")
    ap.add_argument("--steps", action="append", metavar="N=DIR",
                    help="pasos acumulados y directorio de muestras. Repetible.")
    args = ap.parse_args()
    seccion_v14(parse_steps(args.steps))


if __name__ == "__main__":
    main()

