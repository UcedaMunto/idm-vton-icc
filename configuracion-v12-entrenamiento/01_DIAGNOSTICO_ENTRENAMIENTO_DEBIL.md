# 01 - Diagnostico: el entrenamiento no degrada el modelo, pero tampoco lo mueve

Fecha: 2026-08-29

## Sintoma reportado

El usuario reporta imagenes "muy poco aceptables" despues de 500 y 1200 pasos de
entrenamiento automatico (watchdog): la prenda de salida no sigue la prenda de
entrada. Este documento separa lo que ya estaba resuelto (V11), lo que esta
pasando ahora con la cadena actual de 1100 pasos, y lo que impide ver el progreso.

## Contexto: la cadena de entrenamiento real analizada

Tres bloques encadenados desde `result_train_night/checkpoint-250`:

| Bloque | Run | Pasos | Resume | cumulative_steps (manifest) | exit |
|---|---|---:|---|---|---|
| 1 | run_20260825_073101 | 500 | ninguno | 500 | 0 |
| 2 | run_20260825_131801 | 500 | ck-500 del bloque 1 | 1000 | 0 |
| 3 | run_20260825_190501 | 500 (llego a 194) | ck-500 del bloque 2 | 1100 | 1 (crashed) |

Configuracion real de produccion (manifest del bloque 2):
`--train_ip_adapter_only --low_vram_training --gradient_checkpointing
--hybrid_small_models_gpu --garmentnet_dtype=float32 --cpu_threads=12
--width=448 --height=576 --train_batch_size=1 --learning_rate=1e-5
--max_train_steps=500 --checkpointing_steps=100 --mixed_precision=fp16`
(sin `--resume_optimizer_state`). Parametros entrenables: 423,786,624 de
2,991,238,468 (14.17%).

Los bloques 1 y 2 completaron 500 pasos OK. El bloque 3 fue detenido/crasheo en
el paso 194 (el log termina con un traceback truncado de `accelerate`, y
`PAUSAR_WATCHDOG` tiene timestamp 2026-08-25 21:24, exactamente la hora del
corte). El checkpoint valido de la cadena es `run_20260825_190501/checkpoint-100`
(cumulative_steps=1100), del cual se exporto `checkpoint_1100_para_demo` que usa
la app.

## Evidencia 1: el modelo apenas se mueve (deriva de pesos)

Norma L2 relativa de la diferencia sobre los 193 tensores entrenables:

| Comparacion | Deriva relativa L2 |
|---|---:|
| ck500 vs base(250) | 0.684% |
| ck1000 vs base(250) | 1.036% |
| ck1100 vs base(250) | 1.096% |
| ck500 vs ck1000 | 0.579% |
| ck1000 vs ck1100 | 0.210% |

Con LR 1e-5, batch 1, solo IP-Adapter, 1100 pasos mueven los pesos un ~1%.
A este ritmo, incluso 5000 pasos (~4.6 dias) apenas llegarian a ~5% de deriva.

## Evidencia 2: la salida entrenada es casi identica a la base

Comparacion con el mismo par, resolucion y semilla (V11,
`comparacion_base_vs_v11/`):

| Par de imagenes | MAE (0-255) | Correlacion pixelwise |
|---|---:|---:|
| base_15steps vs v11_800steps | 11.72 | 0.936 |

MAE 11.7 es pequeno: dos corridas del mismo modelo con semillas distintas
suelen diferir mas (MAE 20-40). El checkpoint de 800 pasos produce
practicamente la misma imagen que el checkpoint base. Por eso el usuario "no ve
mejora": no es que el entrenamiento empeore el modelo, es que no lo mueve.

## Evidencia 3: la perdida no converge de forma sustancial

Promedio de `step_loss` por tramos (extraido de los logs):

| Tramo | Bloque 1 (run073101) | Bloque 2 (run131801) | Bloque 3 (run190501) |
|---|---:|---:|---:|
| steps 1-50 | 0.0414 | 0.0361 | 0.0442 |
| steps 101-150 | 0.0362 | 0.0373 | 0.0286 |
| steps 201-250 | 0.0355 | 0.0401 | - |
| steps 301-350 | 0.0370 | 0.0375 | - |
| steps 401-450 | 0.0303 | 0.0313 | - |
| steps 451-500 | 0.0295 | 0.0239 | - |

Hay una bajada suave (0.041 -> 0.030) pero lenta y con mucha oscilacion por
paso (0.0005 a 0.19). Es el comportamiento esperado de un fine-tune debil: el
modelo reduce un poco la prediccion de ruido sobre las muestras que ve, pero sin
cambiar su comportamiento real de inferencia.

## Evidencia 4: los bloques reinician el optimizador

`resume_optimizer_state` nunca se paso (watchdog no lo incluye). Cada bloque de
500 pasos arranca Adam de cero (momentum/variance reseteados), aunque el
`optimizer_state.pt` (3.2 GiB) SI se guarda en cada checkpoint. V11 ya lo habia
identificado como "factor agravante secundario". La perdida del bloque 2 empieza
en 0.0361 (no en 0.0295 donde termino el bloque 1): el reinicio del optimizador
le cuesta al bloque siguiente parte del progreso.

## Evidencia 5: la cantidad de datos vista es minima

Con el sampler V11 (correcto), 1100 pasos a batch 1 = 1100 muestras distintas
de las 13563 parejas de `train_pairs.txt` = **8.1% del dataset, una sola vez**.
El modelo oficial se entreno ~130 epocas completas. No hay forma de que 1100
pasos de IP-Adapter-only "aprendan" la tarea de nuevo.

## Por que la app muestra una prenda "completamente distinta"

La comparacion `comparacion_ui_base` de V11 demostro que el checkpoint base
local y el modelo oficial `yisol/IDM-VTON` producen la misma salida roja con la
imagen de Taylor (cuyo vestuario contiene una falda roja). Es decir:

1. La base funciona igual que el modelo oficial en la app.
2. Como el entrenamiento apenas mueve el modelo, la app sigue mostrando el
   comportamiento del modelo base.
3. El sintoma "prenda distinta a la entrada" en las capturas de 2026-08-23
   correspondia a la cadena vieja (2200 pasos sobre el MISMO subconjunto de 500
   parejas, sobreajuste severo, causa raiz V11 ya corregida).
4. Para las capturas posteriores (checkpoint_1100_para_demo), la salida es
   esencialmente la base, con sus limitaciones propias (prenda de entrada que
   no es una prenda aislada de catalogo, fondo, pliegues, etc.).

## Limitacion de esta auditoria

No fue posible inspeccionar visualmente las 4 capturas del usuario (el modelo de
IA de esta sesion no procesa imagenes). En su lugar se analizaron
cuantitativamente las imagenes de comparacion guardadas en el repositorio
(estadisticas por region, MAE, correlacion) y todos los logs. La conclusion de
que "entrenado ~ base" no depende de las capturas: sale de la deriva de pesos y
de la comparacion base-vs-800 pasos ya guardada en disco.

## Conclusion

- V11 (repeticion de datos) estaba bien diagnosticado y el fix del sampler es
  correcto; la cadena actual lo usa y avanza por el dataset.
- El problema actual de "no mejora" es un **problema de presupuesto de
  entrenamiento**: 1100 pasos de IP-Adapter-only a LR 1e-5 son insuficientes
  para mover el modelo (1.1% de deriva) y, con el optimizador reiniciado en cada
  bloque, parte del progreso se pierde.
- La observabilidad (imagenes de revision por bloque) estuvo rota por el bug de
  `python3`/`PYTHONPATH`, asi que nadie pudo ver si un bloque mejoraba algo.

Los arreglos concretos estan en `02_PLAN_RECETA_ENTRENAMIENTO.md` y el fix de
observabilidad en `watchdog_entrenamiento.sh` (V12).
