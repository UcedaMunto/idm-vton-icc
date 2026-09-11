# 05 - Plan de tiempo y hardware (con datos medidos)

Fecha: 2026-09-09. **Todas las cifras de ritmo son medidas**, no estimadas: se
derivan de `logs/produccion_continua_v14/train_run_*.log` y de los `.meta`.

## 5.1 Hardware actual

| Recurso | Valor |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, **12.288 MiB (12 GB)**, driver 595.84, CUDA 13.2, tope 170 W |
| CPU | 12 nucleos |
| RAM | 31 GiB totales + 15 GiB swap (el swap trabaja al limite con la app cargada) |
| Disco | 591 GB, **93 GiB libres (84% usado)** |
| SO | Ubuntu 24.04.4 LTS, kernel 6.8 |
| Entorno | conda `idm`, Python 3.10 |

Restriccion estructural: **GPU y app no caben a la vez**. Mientras entrena, la
demo web queda fuera de servicio; el watchdog ademas no lanza entrenamiento si
detecta `gradio_demo/app.py`.

## 5.2 Ritmo real medido (V14, 2026-09-08/09)

Bloques de 500 pasos completados (inicio -> `finished_at` del `.meta`):

| Bloque | run_id | Inicio | Fin | Duracion | s/paso efectivo |
|---|---|---|---:|---:|---:|
| 1 | `run_20260908_220420` | 09-08 22:04:20 | 09-09 04:07:29 | 6 h 03 m 09 s | **43,6** |
| 2 | `run_20260909_041601` | 09-09 04:16:01 | 09-09 10:07:19 | 5 h 51 m 18 s | **42,2** |
| 3 | `run_20260909_100801` | 09-09 10:08:01 | 09-09 15:55:22 | 5 h 47 m 21 s | **41,7** |
| 4 (300 pasos) | `run_20260909_160401` | 09-09 16:04:01 | 09-09 ~19:43 | ~3 h 39 m | **43,8** |

Media: **~42,8 s/paso** -> **~5,95 h por bloque de 500**. El `.log` confirma el
regimen estacionario (~42,85-43,25 s/it en los pasos 264-304). El primer paso
del bloque es mas lento (79 s) por la carga de modelos; a partir de ~10 pasos se
estabiliza.

**Constantes de conversion (usar estas):**

- 1 paso ≈ **43 s**.
- 1 hora ≈ **84 pasos**; 1 dia en continuo ≈ **~2.000 pasos**.
- 1 bloque de 500 pasos ≈ **6,0 h** + ~0,15 h de overhead (carga de modelos +
  guardado de checkpoint + muestra de revision del watchdog).

## 5.3 Escenarios de tiempo (solo esta GPU)

| Objetivo | Pasos | Solo computo | + overhead | Bloques | Dias (24 h) |
|---|---:|---:|---:|---:|---:|
| Smoke | 100 | 1,2 h | 1,3 h | 1 | 0,1 |
| Piloto | 1.000 | 11,9 h | 12,2 h | 2 | 0,5 |
| MVP | 5.000 | 59,7 h | 61,2 h | 10 | 2,6 |
| Produccion corta | 10.000 | 119,4 h | 122,4 h | 20 | 5,1 |
| Produccion estandar | 20.000 | 238,9 h | 244,9 h | 40 | 10,2 |
| Produccion larga | 30.000 | 358,3 h | 367,3 h | 60 | 15,3 |
| 1 epoca (13.563 pares) | 13.563 | 162,0 h | 166,2 h | 28 | 6,9 |
| 2 epocas (27.126 pares) | 27.126 | 324,0 h | 332,3 h | 55 | 13,8 |
| Receta original (130 epocas) | 1.763.190 | 21.060 h | - | 3.527 | **877,5** |

> La receta original del paper (130 epocas sobre 16.382 pares) es **inviable en
> este hardware** (2,4 anos). El presupuesto realista es 5.000-30.000 pasos.

## 5.4 Calendario por fases

### Pista A - I+D no comercial (dataset actual, legal hoy)

| Fase | Trabajo | Duracion |
|---|---|---:|
| A0 | Saneamiento de receta + smoke + gates A/B de 100/500 | 1-2 dias |
| A1 | Piloto 1.000-2.000 pasos + A/B | 1 dia |
| A2 | Produccion R&D 10.000 pasos | 5 dias |
| A3 | Validacion final, export a pipeline y prueba en la app | 1-2 dias |
| **Total A** | | **~8-10 dias** |

### Pista B - Comercial (datos y stack limpios)

| Fase | Trabajo | Duracion |
|---|---|---:|
| B0 | Auditoria legal + decision de licencias/datos | 2-6 semanas (externo) |
| B1 | Reimplementacion/limpieza del stack (clean-room) | 4-12 semanas |
| B2 | Dataset comercial propio: captura + anotacion | 3-8 semanas |
| B3 | Entrenamiento (10.000-30.000 pasos, esta GPU) | 5-15 dias |
| B4 | Validacion, empaquetado y despliegue | 2-4 semanas |
| **Total B** | | **~3-7 meses** |

B3 puede ir en paralelo a B4; B1/B2 son el cuello real.

## 5.5 Cuellos de botella

> Medido en vivo el 2026-09-09 con el entrenamiento corriendo (ver
> `09_CUELLO_DE_BOTELLA_Y_HARDWARE.md`): de los cuatro puntos de abajo, el que
> domina el **ritmo** es el 4 (todo el calculo pesado ocurre en la CPU: GPU al
> 10,9% y 49,3 W), agravado por el 2 (15 GiB de swap al 100% e iowait hasta 31%).
> El 1 (disco) limita la **capacidad**, no la velocidad: su utilizacion medida es
> 31-36% y sus lecturas son casi todas de swap. Con el escritorio activo el paso
> sube de 42,8 a 65-87 s/paso, asi que las tablas de 5.3 son el mejor caso.

1. **Disco (limitante inmediato).** Cada bloque deja ~5,1 GB
   (`trainable_state.pt` 1,7 GB + `optimizer_state.pt` 3,4 GB). Con 93 GiB
   libres y `MIN_FREE_GIB=40` hay margen para **~10 bloques (~5.000 pasos)** sin
   limpieza. Ademas `configuracion-v12-entrenamiento/preservar_checkpoints_pruebas.sh`
   (cron cada 5 min) copia `trainable_state.pt` + `manifest.json` (~1,7 GB por
   checkpoint) a `result_train_v14/pruebas_checkpoints/`, que ya ocupa 27 GB.
   **Plan**: purgar `run_*` antiguos de `produccion_continua` (solo hace falta
   el ultimo para encadenar) y podar `pruebas_checkpoints`. Ver doc 06.
2. **RAM/swap.** 31 GiB con 15 GiB de swap ya en uso. `--resume_optimizer_state`
   consume ~3,4 GiB extra. No ejecutar la app ni navegadores pesados en paralelo.
3. **GPU unica.** App y entrenamiento se excluyen mutuamente.
4. **CPU-side.** Con `--hybrid_small_models_gpu`, el UNet entrenable y el
   GarmentNet viven en CPU: ahi esta la mayor parte del tiempo por paso.

## 5.6 Aceleraciones sobre el hardware actual

| Medida | Ahorro estimado | Riesgo |
|---|---|---|
| Subir `--checkpointing_steps` de 100 a 250 | Menos guardados (1,7+3,4 GB c/u) | Menos granularidad para A/B |
| Podar `pruebas_checkpoints` | Libera disco, no tiempo | Perder checkpoints historicos |
| `--use_8bit_adam` | Menos RAM de optimizador | Cambia la numerica (validar) |
| Cachear latentes/text-embeddings | Alto (evita VAE/CLIP por paso) | Requiere desarrollar cache |
| Mover el UNet entrenable a GPU | Alto | No cabe en 12 GB (OOM) |

## 5.7 Comparativa de hardware alternativo (estimacion orientativa)

| Hardware | VRAM | Multiplicador aprox. | 10.000 pasos | 30.000 pasos |
|---|---|---:|---:|---:|
| RTX 3060 (actual) | 12 GB | 1x | ~5,1 dias | ~15,3 dias |
| RTX 4090 | 24 GB | 4-6x | ~1 dia | ~2,5-3,8 dias |
| A100 80 GB | 80 GB | 5-8x | ~0,6-1 dia | ~1,9-3 dias |
| H100 80 GB | 80 GB | 8-12x | ~0,4-0,6 dia | ~1,3-1,9 dias |

> Multiplicadores orientativos (batch mayor, sin offload a CPU, mas ancho de
> banda). Coste cloud tipico A100 80 GB: ~3-4 USD/h -> 10.000 pasos ≈ 12-24 h
> ≈ **40-90 USD**; 30.000 pasos ≈ 100-250 USD. Cifras a confirmar con el
> proveedor. Con datos propios, **alquilar GPU para B3 suele salir mas barato
> que 15 dias de esta maquina**.

## 5.8 Reproducir el calculo

`python3 plan_tiempo_v14.py` imprime estas tablas a partir de los parametros
medidos. Se pueden ajustar con variables de entorno (ver el propio script y
`config_v14_comercial.env`): `SECONDS_PER_STEP`, `OVERHEAD_H_PER_BLOCK`,
`STEPS_PER_BLOCK`, `TRAIN_PAIRS`.

