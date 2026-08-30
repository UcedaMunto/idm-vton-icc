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

## Medicion de velocidad real (bloque V12, 2026-08-29)

- Paso 1 (arranque en frio, incluye carga de modelos): 91s.
- Pasos 2-5 en regimen: ~62-96s (media ~70s/paso), vs los 41-58s/paso medidos
  por V10. La diferencia se debe a presion de memoria: el sistema tiene
  31 GiB de RAM y el entrenamiento requiere ~25 GiB RSS; con el swap al 100%
  (15/15 GiB) los pasos van mas lentos pero estables.
- Estimacion del bloque de 500 pasos: ~9.7 horas. Si se quiere recuperar
  velocidad, cerrar aplicaciones con uso alto de RAM (navegadores, IDE) antes de
  cada bloque; el watchdog comprueba `MIN_AVAILABLE_RAM_GIB` (3 GiB) pero no
  controla la presion de swap.

## Pruebas con resultados iniciales (sin esperar al modelo final)

El entrenamiento NO se detiene solo: el watchdog encadena bloques de 500 pasos
indefinidamente. Para probar resultados iniciales:

1. **Preservacion automatica de checkpoints**: el cron de `preservar_checkpoints_pruebas.sh`
   (cada 5 min) copia `trainable_state.pt + manifest.json` de cada checkpoint de
   la corrida activa a `result_train_v10/pruebas_checkpoints/run_*/checkpoint-*`
   ANTES de que el watchdog los borre al terminar el bloque. Desinstalar:
   borrar la linea `preservar_checkpoints_pruebas.sh` del `crontab -l`.
2. **Exportar un checkpoint para la app** (toma ~10 min, usa RAM): 
   `bash configuracion-v12-entrenamiento/exportar_prueba.sh <dir_checkpoint>` y
   apuntar `IDMVTON_MODEL_PATH` del `.env` a la carpeta generada, reiniciar la app.
3. **Comparar sin tocar la app**: 
   `PYTHONPATH=$PWD python3 configuracion-v9-entrenamiento/comparar_calidad_v9.py
   --pretrained_model_name_or_path=result_train_night/checkpoint-250
   --compact_checkpoint=<dir_checkpoint> --data_dir=dataset/DATA_DIR_PREP
   --output_dir=result_train_v10/demos/comparacion --width=576 --height=768`.

### Hitos esperados (bloque 1 V12, 500 pasos a ~45-70s/paso)

| Checkpoint | Cumulative | Hora estimada | Deriva de pesos esperada |
|---|---:|---:|---:|
| checkpoint-100 | 1200 | ~11:10 | ~0.7% (sutil) |
| checkpoint-300 | 1400 | ~14:00 | ~2% |
| checkpoint-500 (fin del bloque) | 1600 | ~16:30-18:00 | ~3.4% (primera version claramente comprobable) |
| despues de 2 bloques | 2100 | ~22:30-02:00 | ~7% |

Para pausar cuando se quiera evaluar: `touch "entrenamiento continuo/PAUSAR_WATCHDOG"`.

## Decision tras la primera prueba con la app (2026-08-29)

El usuario probo la app con el checkpoint cumulative 1800 y reporto que "el
error persiste" (la prenda de salida sigue sin parecerse a la de entrada).

Deduccion clave: el modelo 1800 difiere solo 0.3-0.6% del modelo anterior (1100),
que a su vez era ~base. Por lo tanto, si con el modelo 1800 "el error persiste",
**el modelo base ya tenia ese comportamiento con esas entradas**. No es un
defecto introducido por el entrenamiento, ni algo que "mas pasos" de la receta
actual (IP-Adapter-only, GarmentNet congelado) vaya a cambiar radicalmente.

### Experimentos para confirmar el origen (en orden de costo)

1. **Probar la base directamente en la app** (1 min, costo cero): poner
   `IDMVTON_MODEL_PATH=/home/uceda/Documents/IDM-VTON/result_train_night/checkpoint-250`
   en `.env`, reiniciar la app, misma prenda de entrada.
   - Si la salida es igual a la del 1800 -> el error es del modelo base con ese
     dominio de entrada (desajuste de dominio o condicionamiento), no falta de
     entrenamiento.
   - Si la salida cambia -> el entrenamiento SI esta moviendo el comportamiento
     y "mas pasos" es un camino valido.
2. **A/B controlado con pares validos del dataset** (cerrar la app, ~10 min):
   `comparar_calidad_v9.py` base vs 1800 sobre un par valido (p.ej. 048393_0,
   cuya prenda SI es una prenda aislada; 048392_0 NO es valido: su "prenda" es
   una foto de persona).
3. **Preprocesar la prenda de entrada** (sin entrenar): si las fotos reales
   traen fondo/percha/dobladuras, aislar la prenda y poner fondo blanco antes de
   subirla. El modelo fue entrenado con prendas aisladas en fondo blanco.

### Caminos segun el resultado

- Si el error persiste incluso con prendas aisladas de catalogo -> el problema
  es de receta/condicionamiento; evaluar (a) incluir GarmentNet en el
  entrenamiento, (b) LoRA sobre el UNet completo, o (c) dataset custom con las
  prendas del usuario (adaptacion de dominio).
- Si el error aparece solo con fotos reales con fondo -> atacar el dominio:
  preprocesado de prenda y/o dataset custom con las fotos reales del usuario.
- Si la salida cambia entre base y 1800 -> seguir acumulando pasos y reevaluar
  en 5000-10000.

Estado actual: entrenamiento DETENIDO (watchdog pausado), app corriendo con el
checkpoint 1800.

