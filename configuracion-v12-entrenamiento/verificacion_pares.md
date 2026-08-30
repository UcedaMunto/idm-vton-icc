# Verificacion de pares del dataset (2026-08-29)

Objetivo: tomar 3 pares de `test_pairs.txt` y verificar que prenda (`cloth`),
persona (`image`), mascara (`agnostic-mask`) y DensePose (`image-densepose`)
estan correctamente colocadas y correspondidas por nombre.

## Pares elegidos (validos)

| Par | Prenda (cloth) | Item (annotacion) | obj_frac | Persona | 4 canales |
|---|---|---|---|---|---|
| `048393_0.jpg` | marron, central | Shirt / Short Sleeve / Round Neck | 0.34 | full-body | ✓ |
| `048397_0.jpg` | durazno/beige, central | Shirt / Short Sleeve / Round Neck | 0.43 | full-body | ✓ |
| `048400_0.jpg` | azul oscuro, central | Shirt / Short Sleeve / Round Neck | 0.46 | full-body | ✓ |

Lamina visual para confirmar a ojo:
`configuracion-v12-entrenamiento/verificacion_pares_3.png`
(filas = par, columnas = Prenda | Persona | Mascara | DensePose).

## Criterios usados (no hay vision humana en esta sesion)

- **Prenda valida**: la imagen `cloth` muestra un objeto central aislado sobre
  fondo claro, sin tocar los bordes superior/por completo (bbox R0 > 0.1) y con
  color de prenda reconocible (no gris/negro uniforme ni casi-vacio).
- **Persona valida**: `image` con objeto que ocupa la altura completa (full-body).
- **Correspondencia**: los 4 archivos comparten el mismo nombre de par (por
  construccion del dataset, formato `imagen imagen` en `test_pairs.txt`).

## Pares INVALIDOS detectados en el dataset (importante)

| Par | Problema |
|---|---|
| `048392_0.jpg` | la "prenda" es una foto de persona completa (obj_frac 0.69, bbox 0-1) |
| `048402_0.jpg` | la "prenda" esta casi vacia (obj_frac 0.01, RGB gris) |
| `048406_0.jpg` | la "prenda" esta casi vacia (obj_frac 0.02, RGB blanco/gris) |

> No usar esos pares para evaluar calidad de prenda. `048392_0` ya habia sido
> detectado en V11 como caso invalido.

## Uso recomendado

Estos 3 pares validos son idoneos para el A/B controlado base-vs-entrenado
(`comparar_calidad_v9.py`): misma prenda, misma persona, misma semilla.
