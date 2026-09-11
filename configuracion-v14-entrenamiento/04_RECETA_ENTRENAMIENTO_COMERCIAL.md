# 04 - Receta de entrenamiento (Pista A y Pista B)

Fecha: 2026-09-09. Recoge la evidencia de V10-V13 y la traduce en una receta
utilizable, primero en I+D (Pista A, dataset actual) y luego en comercial
(Pista B, datos propios). Los flags citados existen en `train_xl.py`.

## 4.1 Lecciones de la investigacion previa (no repetir errores)

1. Partir **siempre** del modelo oficial, nunca de un checkpoint de una cadena
   previa (`result_train_night` no era el oficial; ver V13/04).
2. El fine-tune actual (solo IP-Adapter + augmentacion de color agresiva)
   **degrada** la transferencia de color de la prenda en el A/B controlado.
3. Entrenar solo el IP-Adapter (14,17% de parametros) no da fidelidad de prenda:
   el **GarmentNet** (unet_encoder), que aporta textura/geometria, esta congelado.
4. `hue=0.5` (±180 grados) ensena invariancia al color -> candidato al
   desplazamiento de color. Reducirlo.
5. Antes de "mas pasos", hay que demostrar con un A/B que la receta mejora.

## 4.2 Que entrenar

| Opcion | Que se actualiza | Coste | Fidelidad de prenda | Cuando usarla |
|---|---|---|---|---|
| C1 | Solo IP-Adapter (`--train_ip_adapter_only`) | Bajo | Baja (demostrado) | No recomendada para produccion |
| C2 | IP-Adapter + GarmentNet | Medio | Alta | **Recomendada** para fidelidad |
| C3 | IP-Adapter + LoRA sobre el UNet | Medio | Alta + flexible | Alternativa si hay VRAM |

En la RTX 3060 de 12 GB, C2/C3 requieren `--gradient_checkpointing`
y `--low_vram_training` (ya activos en el watchdog). Si no cabe, subir
`--gradient_accumulation_steps` antes que el batch.

## 4.3 Hiperparametros recomendados

| Parametro | Valor | Nota |
|---|---|---|
| `--train_batch_size` | 1 | Limitado por 12 GB VRAM |
| `--gradient_accumulation_steps` | 4-8 | Emula batch 4-8 sin mas VRAM |
| `--learning_rate` | 1e-5 a 2e-5 | 5e-5 es riesgo de sobreajuste; 1e-5 apenas mueve |
| `--max_grad_norm` | 1.0 | Estabilidad |
| `--weight_decay` | 1e-2 | Default del proyecto |
| `--num_tokens` | 16 | Tokens de IP-Adapter |
| `--seed` | 42 | Reproducibilidad (barajado resumible V11) |
| `--checkpointing_steps` | 100 | Permite A/B por checkpoint |
| `--resume_optimizer_state` | activo | Continuidad real entre bloques |
| `--garmentnet_dtype` | float32 | Evita el bug de rendimiento (V9/V10) |
| `--hybrid_small_models_gpu` | activo | Mejor tiempo/paso medido |
| `--cpu_threads` | 12 | Revierte `OMP_NUM_THREADS=1` de accelerate |

## 4.4 Augmentacion (valores REALMENTE en efecto)

Importante: el watchdog **no** pasa ningun `--color_jitter_*`; por tanto se usan
los **defaults de `train_xl.py`** (lineas 344-348). Para cambiarlos hay que
editar `entrenamiento continuo/watchdog_entrenamiento.sh` (no hay variable de
entorno para ellos) o lanzar un bloque manual.

| Parametro | En efecto hoy (defaults) | Recomendado | Motivo |
|---|---|---|---|
| `--color_jitter_prob` | 0.5 | 0.3 | Menos invariancia al color |
| `--color_jitter_brightness` | 0.2 | 0.1 - 0.2 | Suave |
| `--color_jitter_contrast` | 0.2 | 0.1 - 0.2 | Suave |
| `--color_jitter_saturation` | 0.2 | 0.1 - 0.2 | Suave |
| `--color_jitter_hue` | 0.1 | **0.0 - 0.05** | Hue alto = cambio de color de la prenda |

Contexto historico: el proyecto original usaba `hue=0.5` (±180 grados); V13 lo
bajo a 0.1 y ese valor quedo como default del codigo. Sigue siendo un candidato
fuerte al desplazamiento de color, por lo que se recomienda probar `hue=0`.
Cambiar **una sola variable** por experimento y validar con el A/B de 4.6.

## 4.5 Resolucion

- Entrenamiento actual: **448x576**; app en LOW_RAM: **576x768**; oficial: 768x1024.
- Recomendacion: alinear entrenamiento e inferencia al menos a **576x768**; si
  la VRAM lo permite, 768x1024 (calidad del modelo oficial).
- Revisar `compute_time_ids()` (`(height, height)` en el original): deberia ser
  `(height, width)`. Es un ajuste de bajo riesgo y mejora la coherencia.

## 4.6 Puertas de calidad A/B (obligatorias antes de producir mas)

Protocolo (ya validado en V13/04), con `comparar_calidad_v9.py`:

- Par de referencia: `048400_0` (prenda de catalogo AZUL), seed 42, 448x576,
  15 pasos, 1 par. Comparar siempre contra el **oficial** y contra el
  checkpoint anterior.
- Metrica: % de pixeles del torso azul / rojo / oscuro (ver `ab_analisis.py`).
- Criterio de aceptacion: **torso azul > 50%** y sin artefactos rojo/magenta.

| Hito | Pasos | Que valida |
|---|---|---|
| Smoke | 1 | Loss finito, LR correcto en manifest, arranque sin error |
| Gate 1 | 100 | No degrada vs oficial en el par de control |
| Gate 2 | 500 | Mejora o iguala al oficial |
| Gate 3 | cada 1.000 | Sin sobreajuste; mejora sostenida |

Si un gate falla -> **pausar** (`PAUSAR_WATCHDOG`), no acumular pasos.

## 4.7 Anti-sobreajuste

- Reservar el `test` (1.800 / 1.491 limpios) para validacion; no entrenar con el.
- Vigilar `step_loss` (V14: rango tipico 0,0001-0,12; media ~0,03-0,05).
- Vigilar deriva de pesos L2 vs base (V13 midio 0,68%->1,39% en 1.800 pasos).
- Detener cuando la deriva crezca sin mejorar los gates.
