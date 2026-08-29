# Version 12: diagnostico de por que el entrenamiento no mejora las imagenes

Fecha de creacion: 2026-08-29

## Estado

`DIAGNOSTICO_COMPLETO_CON_EVIDENCIA`. No se relanza el entrenamiento de produccion
todavia; primero se corrige la observabilidad (el watchdog no generaba ninguna
imagen de revision) y luego se aplica el plan de receta de una variable a la vez
(protocolo V6).

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
- `watchdog_entrenamiento.sh`: nueva variable `LEARNING_RATE` (default 1e-5,
  sin cambio de comportamiento) para permitir experimentos de LR sin editar el
  script.

## Pendiente / proximos pasos (protocolo V6, una variable a la vez)

Ver `02_PLAN_RECETA_ENTRENAMIENTO.md`. En orden:

1. Confirmar la receta base (V12): resume_optimizer + LR 1e-5, correr 1 bloque y
   medir continuidad (loss no debe volver a subir al inicio del bloque 2).
2. (Experimento 1) subir LR a 2e-5 o 5e-5; validar con set fijo de parejas
   validas, a resolucion de la app (576x768).
3. (Experimento 2) si hace falta mas velocidad por paso, probar 384x512.
4. Recien despues evaluar si conviene tocar GarmentNet o cambiar la resolucion de
   la app para alinearla con entrenamiento.
