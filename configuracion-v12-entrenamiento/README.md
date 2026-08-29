# Version 12: diagnostico de por que el entrenamiento no mejora las imagenes

Fecha de creacion: 2026-08-29

## Estado

`IMPLEMENTADA_Y_EN_EJECUCION`. Diagnostico completo con evidencia, correcciones
aplicadas al watchdog y cadena de produccion relanzada el 2026-08-29 con la
receta V12 (resume_optimizer + LR 5e-5) desde el checkpoint-100 (1100 pasos
acumulados).

## Resumen ejecutivo

El entrenamiento actual **no esta "roto" en el sentido de crashear o dar loss
NaN**: corre, pierde suave (0.041 -> 0.029 promedio en el bloque 1) y produce
checkpoints validos. El problema es que **mueve tan poco el modelo que la salida
es practicamente identica a la base**, y lo que el usuario ve en la app es
(con muy pequeno margen) el comportamiento del modelo base, no el de un modelo
aprendido:

- Deriva relativa L2 de los pesos entrenables: 0.68% (500 pasos), 1.04%
  (1000), 1.10% (1100). Ver `01_DIAGNOSTICO_ENTRENAMIENTO_DEBIL.md`.
- La imagen generada con el checkpoint de 800 pasos vs la del checkpoint base
  difiere solo en MAE 11.7/255 (correlacion 0.936): visualmente casi iguales.
- El protocolo original de IDM-VTON entrena ~130 epocas sobre las 16382 parejas
  (~350k updates). 1100 pasos es el **0.3%** de ese presupuesto.
- Solo se entrenan las capas IP-Adapter + conv_in (14.17% de los parametros,
  `--train_ip_adapter_only`); el GarmentNet (unet_encoder), que aporta la
  textura/geometria de la prenda, esta congelado y no se adapta.
- Ademas, cada bloque de 500 pasos **reinicia el optimizador** (nunca se paso
  `--resume_optimizer_state`), aunque los `optimizer_state.pt` (3.2 GiB) SI se
  guardan en cada checkpoint.

## Bugs operativos encontrados (independientes del entrenamiento)

1. **El watchdog nunca genero imagenes de revision**: llama a
   `comparar_calidad_v9.py` con `python3` y sin `PYTHONPATH` al repo, y el script
   importa `from src...`; fallo siempre con
   `ModuleNotFoundError: No module named 'src'`. Verificado en
   `entrenamiento continuo/imagenes_revision/run_20260825_131801/generar_muestra.log`.
2. **La verificacion V11 del modelo entrenado quedo incompleta**:
   `verificacion_final/entrenado_1100/` esta vacio y su log se corta en el paso
   8/20 de denoising; nunca se comparo visualmente el modelo entrenado.
3. **El caso de prueba `048392_0` usado para comparar es invalido**: su
   "prenda" (`test/cloth/048392_0.jpg`) es una foto de persona completa, no una
   prenda aislada.
4. **Desajuste de resolucion entrenamiento vs app**: el entrenamiento corre a
   448x576 y la app a 576x768 (`TARGET_WIDTH/HEIGHT` en LOW_RAM_MODE).
5. `compute_time_ids()` en `train_xl.py` usa `(args.height, args.height)` como
   target_size en vez de `(args.height, args.width)` (heredado del original;
   afecta condicionamiento posicional, no es la causa principal).

## Cambios aplicados en V12

- `watchdog_entrenamiento.sh`: correccion del bug de la revision visual
  (`PYTHONPATH` al repo + conda env). Sin impacto matematico.
- `watchdog_entrenamiento.sh`: nueva variable `RESUME_OPTIMIZER_STATE=1` (por
  defecto activa) que pasa `--resume_optimizer_state` a `train_xl.py`, usando el
  `optimizer_state.pt` que ya se persistia. Es el unico cambio matematico y
  corresponde al "factor agravante secundario" documentado en V11.
- `watchdog_entrenamiento.sh`: nueva variable `LEARNING_RATE` con **default 5e-5**
  (RECETA V12 PRODUCCION). La decision de subir de 1e-5 se basa en el
  diagnostico de `01_DIAGNOSTICO_ENTRENAMIENTO_DEBIL.md`: a 1e-5 el modelo se
  mueve ~0.7% por bloque de 500 pasos (sin efecto visible); 5e-5 es el valor
  recomendado para fine-tune de solo IP-Adapter con batch 1.

## Ejecucion de produccion (2026-08-29)

- Reanudada la cadena desde `run_20260825_190501/checkpoint-100`
  (cumulative_steps=1100) con la receta V12: `--resume_optimizer_state`
  + `--learning_rate=5e-5`, resto de parametros V10 (Tier1 + float32 +
  cpu_threads=12, 448x576, 500 pasos por bloque).
- Validacion previa: smoke test de 1 paso con resume de optimizador confirmo
  `optimizer state resume enabled`, `cumulative_steps=1101` y checkpoint valido.
- Meta de pasos acumulados: 5000-10000 (2.3-4.6 dias de encadenamiento del
  watchdog). Evaluar calidad con el set fijo del plan cuando se alcance.
- Criterios de parada: revisar las imagenes de revision que ahora genera el
  watchdog (`imagenes_revision/`) y pausar (`PAUSAR_WATCHDOG`) si la perdida
  sube de forma sostenida o la calidad visual empeora.

## Pendiente / proximos pasos (protocolo V6, una variable a la vez)

Ver `02_PLAN_RECETA_ENTRENAMIENTO.md`. En orden:

1. Monitorear la continuidad del primer bloque V12: el `step_loss` debe arrancar
   donde termino la cadena previa (~0.02-0.04, sin saltos sostenidos) y las
   imagenes de revision (ahora funcionales) no deben mostrar artefactos nuevos.
2. Dejar acumular 5000-10000 pasos y validar calidad con el set fijo de parejas
   validas, a resolucion de la app (576x768).
3. Si 5e-5 resultara inestable, bajar a 2e-5 y observar (nunca mezclar con otro
   cambio).
4. Recien despues evaluar si conviene tocar GarmentNet o cambiar la resolucion de
   la app para alinearla con entrenamiento.
