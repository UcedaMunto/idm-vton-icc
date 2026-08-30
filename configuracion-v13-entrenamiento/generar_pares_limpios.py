#!/usr/bin/env python3
"""V13: genera {phase}_pairs_clean.txt descartando SOLO prendas inequivocamente
malas, sin tocar los datos originales:
  - PERSONA: foto de persona usada como prenda (alto contenido de piel).
  - UNIFORME/VACIA: imagen sin estructura (std muy bajo y sin bordes) -> no es prenda.
Se CONSERVAN prendas blancas/claras (tienen bordes/silueta) y oscuras.
Escribe solo texto; no modifica imagenes.
"""
import os
import numpy as np
from PIL import Image
from collections import Counter

DATASET = os.environ.get('DATA_DIR', '/home/uceda/Documents/IDM-VTON/dataset/DATA_DIR_PREP')
SKIN_PERSON = 0.15
OBJ_PERSON = 0.50
STD_EMPTY = 10.0
EDGE_EMPTY = 0.006

def es_valida(img):
    a = np.asarray(img.convert('RGB').resize((96, 128)), dtype=np.float32)
    gray = a.mean(axis=2)
    frac = (gray < 228).mean()
    std = float(gray.std())
    gy, gx = np.gradient(gray)
    edge = float((np.abs(gx) + np.abs(gy) > 40).mean())
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    skin = float(((R > G) & (G > B) & (R > 60) & (R < 250) & ((R - B) > 12)).mean())
    if skin > SKIN_PERSON and frac > OBJ_PERSON:
        return False, f'persona skin={skin:.2f}'
    if std < STD_EMPTY and edge < EDGE_EMPTY:
        return False, f'uniforme std={std:.1f} edge={edge:.3f}'
    return True, f'ok obj={frac:.2f} std={std:.1f} skin={skin:.2f}'

def filtrar(phase):
    pairs = os.path.join(DATASET, f'{phase}_pairs.txt')
    out = os.path.join(DATASET, f'{phase}_pairs_clean.txt')
    lineas = [l.strip() for l in open(pairs) if l.strip()]
    validos, invalidos = 0, []
    with open(out, 'w') as fo:
        for line in lineas:
            im, cl = line.split()[:2]
            p = os.path.join(DATASET, phase, 'cloth', cl)
            if not os.path.isfile(p):
                invalidos.append((im, 'no existe')); continue
            ok, motivo = es_valida(Image.open(p))
            if ok:
                fo.write(line + '\n'); validos += 1
            else:
                invalidos.append((im, motivo))
    c = Counter(m.split()[0] for _, m in invalidos)
    print(f'== {phase}: {len(lineas)} -> {validos} validos, {len(invalidos)} descartados')
    print('   motivos:', dict(c))
    return validos, len(invalidos)

if __name__ == '__main__':
    filtrar('train')
    filtrar('test')
