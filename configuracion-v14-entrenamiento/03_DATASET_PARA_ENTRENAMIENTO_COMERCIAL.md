# 03 - Dataset para entrenamiento comercial

## 3.1 El dataset actual (medido el 2026-09-09)

| Elemento | Valor |
|---|---|
| Raiz | `/home/uceda/Documents/IDM-VTON/dataset` |
| Tamano total en disco | **109 GiB** |
| Carpeta usada por el entrenamiento | `dataset/DATA_DIR_PREP` |
| Pares train | **13.563** (`train_pairs.txt`) |
| Pares train "limpios" | **11.349** (`train_pairs_clean.txt`) |
| Pares test | **1.800** (`test_pairs.txt`) / 1.491 limpios |
| Estructura por fase | `train|test/{cloth,image,agnostic-mask,image-densepose}` |
| Licencia | VITON-HD, CC BY-NC 4.0, **solo investigacion** (`shadow2496/VITON-HD`) |

**El dataset actual NO puede usarse para un modelo comercial.** Se conserva
porque es excelente para desarrollar y validar la receta (Pista A) antes de
invertir en datos con licencia comercial (Pista B).

## 3.2 Formato exacto que exige el codigo (`train_xl.py`)

El cargador `VitonHDDataset` (linea 34) lee, por cada par `(persona, prenda)`:

```
DATA_DIR_PREP/
  train/
    cloth/<cloth_name>.jpg            # prenda aislada
    image/<image_name>.jpg            # persona objetivo
    agnostic-mask/<image_name>_mask.png   # mascara agnostica
    image-densepose/<image_name>.jpg  # densepose alineada a la persona
  train_pairs.txt          # lineas: "<image_name> <cloth_name>"
  train_pairs_clean.txt    # opcional; si existe, se usa ESTE (linea 113-114)
  test_pairs.txt / test_pairs_clean.txt
  vitonhd_train_tagged.json   # anotacion de prenda -> caption
  vitonhd_test_tagged.json
```

Detalles verificados en codigo:

- `cloth` se abre por `c_name`; `image` por `im_name`; la mascara por
  `im_name.replace('.jpg','_mask.png')`; densepose por `im_name`.
- El caption se construye como `"model is wearing " + cloth_annotation` y
  `"a photo of " + cloth_annotation`, con la anotacion tomada de
  `vitonhd_<phase>_tagged.json` (si falta, usa `"shirts"`).
- El loader prefiere `*_pairs_clean.txt` cuando existe.
- Resolucion de entrenamiento actual del watchdog: **448x576** (`--width`,
  `--height`); el valor del proyecto original es 768x1024.

Para reutilizar este codigo con un dataset propio **basta replicar esa
estructura y esos nombres de archivo/columna**. Es lo mas barato en ingenieria.

## 3.3 Requisitos de un dataset comercial

1. **Licencia clara y permisiva** (propia o con clausula comercial explicita).
2. **Consentimiento/cesion de derechos de imagen** de las personas: guardar un
   *release* firmado por cada sujeto. En Europa/Espana afecta al RGPD.
3. **Derechos sobre las prendas**: fotografia propia del producto o permiso del
   titular de la marca.
4. **Autorizacion de la herramienta de anotacion** que genere pose/densepose.
5. Trazabilidad: registrar por imagen el origen, fecha y licencia (procedencia).

## 3.4 Opciones de datos con licencia comercial

| Opcion | Coste | Control | Riesgo legal | Notas |
|---|---|---|---|---|
| **A. Datos propios** (sesion fotografica prenda+persona) | Medio | Total | Bajo | Requiere consentimiento; el mejor camino a largo plazo |
| **B. Acuerdo con una marca/retailer** (sus catalogo+modelos) | Variable | Alto | Bajo | Datos reales de catalogo, ideal para el dominio del usuario |
| **C. Datasets publicos con licencia permisiva** | Bajo | Medio | Medio | Hay que auditar licencia pieza a pieza; escasean en try-on |
| **D. Datos sinteticos generados** | Bajo | Alto | Bajo | Util como complemento; riesgo de sesgo del generador |

## 3.5 Herramientas de anotacion: que es reutilizable

| Necesidad | Herramienta actual | Licencia | Sustituto comercial |
|---|---|---|---|
| Mascara agnostica | SCHP (`parsing_*.onnx`) | MIT (codigo) | Reentrenar/verificar pesos, o parser propio |
| Densepose (input) | DensePose R-CNN | CC BY-NC | Modelo de pose propio con licencia permisiva |
| Pose / keypoints | OpenPose | No comercial | Alternativa Apache-2.0 (p.ej. MediaPipe) o modelo propio |

**Importante:** no basta con cambiar el dataset de entrenamiento. Si para
**producir** el densepose de las imagenes de entrenamiento (o de inferencia) se
usa DensePose/OpenPose, esas herramientas tambien contaminan la cadena. Hay que
sustituirlas por alternativas permisivas o generar el input con un modelo propio.

## 3.6 Cuanto dato hace falta (orientativo)

| Objetivo | Pares minimos | Comentario |
|---|---:|---|
| Prueba de concepto (receta) | 500 - 2.000 | Valida pipeline y gates |
| MVP comercial | 5.000 - 15.000 | Equiparable al dataset actual |
| Calidad de produccion | 20.000 - 100.000 | Mas variedad de prendas/cuerpos/poses |

Mas importante que la cantidad es la **cobertura de dominio**: tipos de prenda,
tonos de piel, cuerpos, poses, iluminacion y fondos. Un dataset pequeno y bien
etiquetado rinde mejor que uno grande y sesgado.

## 3.7 Checklist del dataset comercial

- [ ] Cada imagen tiene licencia documentada y trazable (procedencia + fecha).
- [ ] Cada persona tiene consentimiento/cesion de derechos de imagen (RGPD si aplica).
- [ ] Cada prenda tiene derechos de uso comercial confirmados.
- [ ] La herramienta de anotacion (pose/parsing) es comercialmente usable.
- [ ] Estructura identica a 3.2 (`cloth/`, `image/`, `agnostic-mask/`,
      `image-densepose/`, `*_pairs.txt`, `vitonhd_*_tagged.json`).
- [ ] Sin imagenes de menores sin autorizacion expresa.
- [ ] Split train/test disjunto por persona (no filtrar identidades).
- [ ] Curaduria visual: descartar pares invalidos (persona como prenda, imagenes
      vacias). Criterios en `configuracion-v13-entrenamiento/03`.
