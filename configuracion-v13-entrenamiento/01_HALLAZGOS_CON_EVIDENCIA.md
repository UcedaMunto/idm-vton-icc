# 01 - Hallazgos con evidencia (investigacion v11 -> v13)

Fecha: 2026-08-29. Todo lo que sigue fue **medido o verificado** en esta
maquina, no supuesto.

## A. Sintoma reportado (usuario + ChatGPT)

Con el modelo entrenado (probado en la app con el checkpoint cumulative 1800):
- La **geometria** de la prenda no se transfiere (termina como camiseta/sueter generico).
- El **color** se desplaza hacia rojo/naranja (ej. camiseta azul de entrada -> salida naranja/roja).
- El modelo aprende **textura/ruido** pero no la identidad real de la prenda.
- La mascara (upper_body: torso, hombros, mangas; conserva cabeza/manos/shorts) es razonable.

## B. El modelo apenas se mueve (clave)

Deriva relativa L2 de los tensores entrenables (193):

| Comparacion | Deriva |
|---|---:|
| ck500 vs base(250) | 0.684% |
| ck1000 vs base(250) | 1.036% |
| ck1100 vs base(250) | 1.096% |
| **ck1800 (V12, LR 5e-5) vs base** | **1.39%** |
| ck1800 vs modelo de la app (1100) | 0.325% total (IP-adapter 0.62%, conv_in 0.50%, image_proj 0.48%) |

Salida del modelo 800 vs base por par: MAE 11.7/255, correlacion 0.936 (casi
identicas). Conclusion: **a 1800 pasos el modelo sigue siendo, en la practica,
el modelo base.**

## C. Base == oficial (V11)

La comparacion `comparacion_ui_base` de configuracion-v11 demuestra que el
checkpoint base local y el oficial `yisol/IDM-VTON` producen la misma salida
(imagen de Taylor con falda roja). Como 1800 ~ base, el sintoma es en gran parte
el **comportamiento inherente del modelo** con esas entradas.

## D. Receta debil para fidelidad de prenda

- `--train_ip_adapter_only`: entrenan 423,786,624 de 2,991,238,468 parametros (14.17%).
- GarmentNet (`unet_encoder`) esta **100% congelado**; es la via principal de
  textura/geometria de la prenda. Entrenar solo el IP-Adapter ajusta color/estilo
  semantico, no el detalle.

## E. Emparejamiento de datos (se VERIFICO correcto -> descarta esa causa)

- `train_pairs.txt` / `test_pairs.txt`: formato `imagen imagen` (prendas y
  personas comparten id por nombre).
- Para `000001_0` existen y coinciden por nombre: `cloth/000001_0.jpg`,
  `image/000001_0.jpg`, `agnostic-mask/000001_0_mask.png`,
  `image-densepose/000001_0.jpg`. (El ejemplo del usuario con la ruta de densepose
  duplicada fue un artefacto de copiar/pegar.)
- Verificacion integral: 0 huerfanos, 0 faltantes; mascaras y densepose
  corresponden 1:1 por nombre (13563/13563 train, 1800/1800 test).
- El loader (`VitonHDDataset.__getitem__`) lee `cloth` por `c_name`, `image` por
  `im_name`, `mask` por `im_name.replace('.jpg','_mask.png')` y `densepose` por
  `im_name`. Correcto.

## F. Datos "basura" en el dataset (contenido, no nombres)

- Algunas `cloth` son **fotos de persona** (piel alta: skin>0.15 y obj>0.5).
  Ejemplos verificados: 000000 (skin 0.51), 000123 (skin 0.64), 000083, 000086,
  000097, etc.
- Algunas `cloth` son **imagenes uniformes/vacias** (std<10 y bordes ~0).
  Ejemplo: 000046 (std 8.4, bordes 0.000). El resto de mis "VACIA" (000065,
  000108, 000133) resultaron ser **prendas blancas VALIDAS** (std 12-13.6, con
  bordes), no vacias.
- Pares de test claramente invalidos: 048392 (persona como prenda), 048402,
  048406 (casi vacios).
- Densepose: muestra aleatoria de 40 -> 40/40 con contenido. Caso aislado: el
  densepose de `000001_0.jpg` esta **negro** (meanRGB [2,2,2]) (anomalia, no sistemico).

## G. Augmentacion de color (candidato real)

ColorJitter(brightness=0.5, contrast=0.3, saturation=0.5, **hue=0.5**) aplicado a
imagen Y prenda con p=0.5. `hue=0.5` = +-180 grados (muy agresivo). **Identico al
proyecto original**, asi que no es un cambio nuestro, pero puede ensenar al modelo
a ser casi invariante al color (colabora al desplazamiento de color).

## H. VAE / precision (probablemente NO)

- Entrenamiento carga VAE en float32 (`cpu_dtype`); la inferencia (app) lo carga
  en float16 pero el pipeline hace `force_upcast` a float32 en encode/decode
  (consistente).
- Normalizacion: entrenamiento e inferencia usan `Normalize([0.5],[0.5])`.
- Escala de latentes usa `vae.config.scaling_factor` en ambos.

## I. Checkpoint cargado correctamente (descarta esa causa)

- Export del modelo 1800: **0 claves faltantes**, IP-adapter (140 claves),
  encoder_hid_proj (51), conv_in 13 canales, **config IDENTICO** al
  `checkpoint_1100` que la app ya usaba bien.
- La app carga `demos/prueba_cumulative_1800` sin error (HTTP 200, 16 GB RSS,
  env correcto).

## J. Base de entrenamiento correcta (descarta "entrenar desde SDXL base")

- `pretrained_model_name_or_path=result_train_night/checkpoint-250` (un checkpoint
  del proyecto). Verificado: `in_channels=13` y `encoder_hid_dim_type=ip_image_proj`
  -> es arquitectura IDM-VTON, NO SDXL base (que tiene 4 canales).
- V11 probo que checkpoint-250 se comporta como el oficial.

## K. Learning rate

- El sintoma existia ANTES de subir LR (con el modelo 1100 entrenado a 1e-5).
  -> LR **no es la causa principal**.
- LR 5e-5 (V12) es alto para esta receta y es un **riesgo** de sobreajuste a largo
  plazo (vigilar), aunque a 1800 pasos no desestabilizo (loss sin explotar).

## L. Otras inconsistencias (menores)

- Resolucion: entrenamiento 448x576 vs app 576x768 (LOW_RAM). El oficial entrena a
  768x1024.
- `compute_time_ids()` usa `(args.height, args.height)` en vez de `(height, width)`
  (heredado del original) -> condicionamiento posicional inconsistente.

## M. Bugs corregidos en v11/v12

- **V11**: repeticion de datos (shuffle=False + reinicio) -> `ResumableShuffleSampler`
  + `cumulative_steps` en el manifest del checkpoint.
- **V12**: el watchdog no generaba imagenes de revision (llamaba a
  `comparar_calidad_v9.py` con `python3` sin `PYTHONPATH` -> `ModuleNotFoundError:
  No module named 'src'`). Se corrigio con `PYTHONPATH`. Tambien se activo
  `--resume_optimizer_state` (el `optimizer_state.pt` de 3.2 GiB ya se persistia) y
  se documento la preservacion de checkpoints intermedios y `exportar_prueba.sh`
  (que usaba `python3` del sistema sin diffusers -> corregido a conda python).

