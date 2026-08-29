# Version 11: correccion de la repeticion de datos entre bloques de entrenamiento

Fecha de creacion: 2026-08-24/25

## Estado

`CAUSA_RAIZ_ENCONTRADA_Y_CORREGIDA_VALIDADA`. Origen: el usuario reporto resultados muy
malos en la app de Gradio tras ~2200 pasos de entrenamiento automatico (V10 + watchdog),
con la prenda de salida completamente distinta a la de entrada. Investigacion exhaustiva
(comparando `IDM-VTON-main` original contra `IDM-VTON`) encontro que **cada bloque de 500
pasos entrenaba siempre sobre las mismas primeras 500 parejas del dataset** (de 13563
disponibles), por `shuffle=False` + reinicio de la posicion del dataset en cada proceso —
comportamiento identico en el proyecto original, expuesto por nuestra automatizacion de
bloques cortos encadenados (el script original fue pensado para una sola corrida larga).

## Que hace V11

1. Documenta la causa raiz con evidencia de codigo (`01_CAUSA_RAIZ_REPETICION_DATOS.md`).
2. Corrige `train_xl.py`:
   - Nueva clase `ResumableShuffleSampler`: orden barajado fijo y reproducible (semilla
     `--seed`), rotado para continuar donde quedo la cadena anterior.
   - El manifest de cada checkpoint compacto ahora guarda `cumulative_steps` (pasos
     acumulados de toda la cadena), leido al reanudar para fijar el punto de partida del
     sampler.
3. Valida el mecanismo con una prueba real de 2 bloques encadenados de 3 pasos
   (`02_CORRECCION_MUESTREO_DATASET.md`): confirma lectura/escritura correcta del manifest
   y arranque en el punto correcto.

## Que NO hace V11

- No relanza todavia el entrenamiento de produccion real (la cadena anterior fue borrada el
  2026-08-23 por sospecha de corrupcion; con esta correccion ya se puede reiniciar con
  confianza, pero se espera confirmacion del usuario para hacerlo).
- No cambia nada de la configuracion de rendimiento de V10 (`--hybrid_small_models_gpu
  --garmentnet_dtype=float32 --cpu_threads=12` siguen siendo la combinacion recomendada).
- No modifica el watchdog (`entrenamiento continuo/watchdog_entrenamiento.sh`); la
  correccion es transparente para el, no requiere cambios alli.
- No repite todavia la comparacion de calidad formal (V6) sobre un modelo entrenado con
  esta correccion; eso corresponde a una vez que se acumulen varios bloques nuevos.

## Documentos

1. [01_CAUSA_RAIZ_REPETICION_DATOS.md](01_CAUSA_RAIZ_REPETICION_DATOS.md): sintoma, causa
   raiz con evidencia de codigo (identica en el proyecto original), consecuencia medida, y
   la decision de borrar el entrenamiento acumulado previo.
2. [02_CORRECCION_MUESTREO_DATASET.md](02_CORRECCION_MUESTREO_DATASET.md): el fix aplicado
   (`ResumableShuffleSampler` + `cumulative_steps` en el manifest), validacion real de 2
   bloques encadenados, alcance de la compatibilidad hacia atras, y que no cambia.

3. `comparacion_base_vs_v11/`: comparación controlada guardada en disco entre el checkpoint
  base y V11 (800 pasos acumulados), con el mismo par, resolución y semilla. La prueba mostró
  además que el caso `048392_0` del archivo `test_pairs.txt` no es válido para evaluar una
  prenda superior: imagen y prenda tienen el mismo nombre, pero `test/cloth/048392_0.jpg` es
  una fotografía completa de una persona, no una prenda aislada.

4. `comparacion_ui_base/`: prueba de control en la interfaz con el modelo oficial
  `yisol/IDM-VTON`. La app oficial y el checkpoint local base produjeron la misma salida roja
  al usar la imagen de Taylor, cuyo propio vestuario contiene una falda roja. Esto demuestra
  que el síntoma no puede atribuirse al entrenamiento V11.
