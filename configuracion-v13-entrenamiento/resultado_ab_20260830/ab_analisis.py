#!/usr/bin/env python3
"""A/B: analisis cuantitativo base vs v13 vs v12 sobre par 048400_0.

Metricas por imagen:
- Fraccion de pixeles AZULES (hue azul, sat>0.25, val>0.15)
- Fraccion de pixeles ROJOS (hue rojo, sat>0.25, val>0.15)
- Fraccion "vivos" (sat>0.25): cuanto colorido hay vs neutro/negro
- Hue medio ponderado de los pixeles vivos (para ver el color dominante real)
- SSIM y MAE entre pares de salidas (estructura global)
- SSIM sobre la region de la prenda (mascara del output es la persona; usamos
  la bbox central torso aprox: filas 35%-75%, todo el ancho)
"""
import os, math
import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim
    HAVE_SSIM = True
except Exception:
    HAVE_SSIM = False


def load(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def rgb2hsv(img):
    from colorsys import rgb_to_hsv
    h = np.zeros(img.shape[:2], dtype=np.float32)
    s = np.zeros(img.shape[:2], dtype=np.float32)
    v = np.zeros(img.shape[:2], dtype=np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    v[:] = mx
    with np.errstate(divide="ignore", invalid="ignore"):
        s[:] = np.where(mx > 0, d / np.maximum(mx, 1e-9), 0)
        # hue en grados (0..360)
        h = np.where(d == 0, 0.0, 0.0)
        h = np.where((mx == r) & (d != 0), (60 * ((g - b) / np.maximum(d, 1e-9)) + 360) % 360, h)
        h = np.where((mx == g) & (d != 0), (60 * ((b - r) / np.maximum(d, 1e-9)) + 120) % 360, h)
        h = np.where((mx == b) & (d != 0), (60 * ((r - g) / np.maximum(d, 1e-9)) + 240) % 360, h)
    return h, s, v


def metricas(img, region=None):
    h, s, v = rgb2hsv(img)
    if region is not None:
        h, s, v = h[region], s[region], v[region]
    alive = (s > 0.25) & (v > 0.15)
    n = alive.sum()
    if n == 0:
        return {"alive": 0.0, "azul": 0.0, "rojo": 0.0, "hue_medio": None, "sat_medio": 0.0}
    azul = (((h >= 195) & (h <= 275)) & alive).sum() / n
    rojo = (((h >= 330) | (h <= 20)) & alive).sum() / n
    verde = (((h >= 80) & (h <= 160)) & alive).sum() / n
    # hue medio circular
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
    if HAVE_SSIM:
        s = float(ssim(a, b, channel_axis=2, data_range=1.0))
    else:
        s = None
    return {"mae": mae, "mse": mse, "ssim": s}


def main():
    base_dir = "/tmp/ab_out"
    cloth = load("/tmp/ab/test/cloth/048400_0.jpg")
    base = load(f"{base_dir}/base/048400_0.jpg")
    v13 = load(f"{base_dir}/v13/048400_0.jpg")
    v12 = load(f"{base_dir}/v12/048400_0.jpg")

    # region del torso en el output (la prenda): filas 35-80% del alto
    h0, h1 = int(base.shape[0] * 0.35), int(base.shape[0] * 0.80)
    region = (slice(h0, h1), slice(None))

    print("=" * 78)
    print("COLOR (region del torso en cada salida; para la prenda: imagen completa)")
    print("=" * 78)
    print(f"{'imagen':<10} {'alive%':>7} {'azul%':>7} {'rojo%':>7} {'verde%':>7} {'hue_medio':>10} {'sat_medio':>9}")
    for nombre, img, reg in [
        ("prenda(in)", cloth, None),
        ("base", base, region),
        ("v13", v13, region),
        ("v12", v12, region),
    ]:
        m = metricas(img, reg)
        print(f"{nombre:<10} {m['alive']*100:6.1f}% {m['azul']*100:6.1f}% {m['rojo']*100:6.1f}% "
              f"{m['verde']*100:6.1f}% {str(m['hue_medio']):>10} {m['sat_medio']:9.3f}")

    print()
    print("=" * 78)
    print("SIMILITUD ESTRUCTURAL GLOBAL (toda la imagen) y REGION TORSO")
    print("=" * 78)
    for na, a, nb, b in [("base", base, "v13", v13), ("base", base, "v12", v12), ("v13", v13, "v12", v12)]:
        g = comparar(a, b)
        r = comparar(a, b, region)
        print(f"{na:>4} vs {nb:<4} | MAE global={g['mae']:.4f} SSIM={g['ssim']:.4f} | MAE torso={r['mae']:.4f} SSIM torso={r['ssim']:.4f}")

    print()
    print("=" * 78)
    print("VEREDICTO (heuristica):")
    print("=" * 78)
    cb = metricas(base, region)
    cv13 = metricas(v13, region)
    cv12 = metricas(v12, region)
    for nombre, m in [("base", cb), ("v13", cv13), ("v12", cv12)]:
        if m["azul"] > m["rojo"]:
            print(f"  {nombre}: SALIDA AZUL (azul {m['azul']*100:.0f}% > rojo {m['rojo']*100:.0f}%) -> respeta la prenda")
        else:
            print(f"  {nombre}: SALIDA ROJIZA (rojo {m['rojo']*100:.0f}% >= azul {m['azul']*100:.0f}%) -> sintoma azul->rojo presente")


if __name__ == "__main__":
    main()
