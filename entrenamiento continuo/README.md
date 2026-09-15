# Entrenamiento continuo IDM-VTON (watchdog + cron)

Este paquete crea una tarea programada que se ejecuta cada minuto. Si NO detecta
un proceso de entrenamiento (ni Gradio) activo, lanza automaticamente un bloque
de 500 pasos desde el ultimo checkpoint compacto, usando el modo hibrido GPU/CPU
validado en [configuracion-v10-entrenamiento](../configuracion-v10-entrenamiento/README.md) (Tier1 GPU +
`--garmentnet_dtype=float32` + `--cpu_threads=12`; ~55-58s/paso en regimen
estacionario, 500 pasos ≈ 7.6-8.1 horas), mas las correcciones de muestreo de
dataset/optimizador de V11-V13 y el reinicio desde el modelo oficial de V14.
Como cada bloque dura varias horas, en la practica esto mantiene al equipo
entrenando de forma continua durante todo el dia, encadenando bloques de 500
pasos uno tras otro sin intervencion manual.

Ruta base:
- `/home/uceda/Documents/IDM-VTON/entrenamiento continuo`

Actualizado el 2026-09-10 para usar la configuracion V14 (reinicio desde el
modelo oficial `yisol/IDM-VTON`, cadena en `result_train_v14/produccion_continua`;
ver [configuracion-v14-entrenamiento/10_PLAN_IMPLEMENTACION_V14.md](../configuracion-v14-entrenamiento/10_PLAN_IMPLEMENTACION_V14.md)). El nucleo
de optimizacion GPU/CPU sigue siendo el de V10 (antes usaba V4/V3 con
`--garmentnet_dtype=bfloat16`, el bug de rendimiento corregido en V9/V10).

## Archivos

- `watchdog_entrenamiento.sh`: verifica proceso y relanza entrenamiento si no existe.
- `instalar_cron_watchdog.sh`: instala cron cada minuto para ejecutar el watchdog.
- `desinstalar_cron_watchdog.sh`: elimina la linea de cron del watchdog.
- `PAUSAR_WATCHDOG`: si existe (aunque este vacio), el watchdog no lanza nada. Se crea/borra manualmente.
- `PAUSAR_POR_ERROR`: el watchdog lo crea automaticamente si una corrida falla o no produce un checkpoint valido. Mientras exista, el watchdog no lanza nada nuevo (evita ciclos de crash-loop).

## Como instalar (ya instalado y activo desde 2026-08-22)

```bash
cd "/home/uceda/Documents/IDM-VTON/entrenamiento continuo"
chmod +x watchdog_entrenamiento.sh instalar_cron_watchdog.sh desinstalar_cron_watchdog.sh
./instalar_cron_watchdog.sh
crontab -l
```

Debes ver una linea similar a:

```bash
* * * * * /bin/bash "/home/uceda/Documents/IDM-VTON/entrenamiento continuo/watchdog_entrenamiento.sh" >> "/home/uceda/Documents/IDM-VTON/logs/produccion_continua/cron_watchdog.log" 2>&1
```

## Como funciona

1. Cron ejecuta el watchdog cada minuto.
2. El watchdog usa `flock` para no solaparse consigo mismo si una ejecucion tarda mas de 1 minuto.
3. Si existe `PAUSAR_WATCHDOG` o `PAUSAR_POR_ERROR`, no hace nada.
4. Si hay un proceso `train_xl.py` o Gradio activo, no hace nada (el bloque de 500 pasos en curso sigue corriendo normalmente durante horas; el watchdog solo actua cuando NO hay nada corriendo).
5. Verifica espacio en disco (minimo 90 GiB libres) y RAM disponible (minimo 3 GiB) antes de lanzar.
6. Busca el checkpoint compacto mas reciente en `result_train_v10/produccion_continua` para encadenar; si no hay ninguno, arranca desde `result_train_night/checkpoint-250` sin reanudar pesos entrenables adicionales.
7. Lanza `accelerate launch train_xl.py` con `--hybrid_small_models_gpu --garmentnet_dtype=float32 --cpu_threads=12 --gradient_checkpointing --low_vram_training --train_ip_adapter_only`, 500 pasos, checkpoint cada 100.
8. El lanzamiento ocurre en un subproceso desacoplado (`nohup ... & disown`) que SI espera a que termine el entrenamiento (a diferencia del propio watchdog, que retorna de inmediato). Al terminar, registra el codigo de salida en el `.meta` de esa corrida y, si fallo o no genero el checkpoint esperado, crea `PAUSAR_POR_ERROR` automaticamente y deja de lanzar corridas nuevas hasta que se revise manualmente.
9. Si la corrida termino OK, borra automaticamente los checkpoints intermedios de esa misma corrida (100/200/300/400 si el bloque fue de 500), dejando solo el checkpoint final (el unico que hace falta para encadenar el siguiente bloque). Esto evita llenar el disco con checkpoints ya superados.

## Pausar para probar la app o por mantenimiento

La app de Gradio y el entrenamiento no deben compartir la GPU de 12 GB: pueden
producir un error de memoria. Antes de iniciar la app, pausa el watchdog:

```bash
touch "/home/uceda/Documents/IDM-VTON/entrenamiento continuo/PAUSAR_WATCHDOG"
pkill -f 'train_xl.py'
```

El cron sigue instalado, pero al encontrar `PAUSAR_WATCHDOG` no relanza procesos.
El bloque de 500 pasos que se interrumpe con `pkill` pierde el progreso desde el
ultimo checkpoint guardado (cada 100 pasos), no todo el bloque.

Para reanudar el entrenamiento automatico:

```bash
rm -f "/home/uceda/Documents/IDM-VTON/entrenamiento continuo/PAUSAR_WATCHDOG"
```

(No hace falta ejecutar el watchdog manualmente; el proximo tick de cron, a mas
tardar en 1 minuto, retoma solo.)

## Si el watchdog se pauso por error (`PAUSAR_POR_ERROR`)

1. Revisar el archivo para ver que corrida fallo: `cat "/home/uceda/Documents/IDM-VTON/entrenamiento continuo/PAUSAR_POR_ERROR"`.
2. Revisar el log de esa corrida (`log_file` dentro del archivo anterior) para entender la causa (OOM, error de datos, etc.).
3. Corregir la causa raiz si aplica.
4. Borrar el archivo para que el watchdog vuelva a intentar: `rm -f "/home/uceda/Documents/IDM-VTON/entrenamiento continuo/PAUSAR_POR_ERROR"`.

## Parametros usados por defecto (watchdog, configuracion V14)

- `--max_train_steps=500`
- `--checkpointing_steps=100`
- `--logging_steps=100`
- `--train_batch_size=1`, `--test_batch_size=1`
- `--train_num_workers=1`, `--test_num_workers=1`
- `--width=448`, `--height=576`
- `--low_vram_training --train_ip_adapter_only --gradient_checkpointing`
- `--hybrid_small_models_gpu --garmentnet_dtype=float32 --cpu_threads=12` (optimizacion V10)
- `--resume_optimizer_state` + `--learning_rate=2e-5` (V12/V13)
- `--resume_from_checkpoint=<ultimo checkpoint compacto en result_train_v14/produccion_continua, o ninguno en el primer bloque>`
- Base fija (pesos congelados/arquitectura): `result_train_v14/base_oficial` (snapshot local de `yisol/IDM-VTON`, ver V14; ya no se usa `result_train_night/checkpoint-250`, ver V13)

Todos los valores se pueden sobreescribir con variables de entorno (ver el
encabezado de `watchdog_entrenamiento.sh`), por ejemplo para una prueba manual
con menos pasos: `MAX_TRAIN_STEPS=1 CHECKPOINTING_STEPS=1 bash watchdog_entrenamiento.sh`.

### V12 (2026-08-29)

- `RESUME_OPTIMIZER_STATE=1` (nuevo, por defecto activado): pasa
  `--resume_optimizer_state` a `train_xl.py`, cargando el `optimizer_state.pt`
  (3.2 GiB) que ya se persistia en cada checkpoint. Elimina el reinicio del
  optimizador entre bloques (factor agravante documentado en V11). Costo extra
  de RAM ~3.4 GiB al reanudar. Para desactivar: `RESUME_OPTIMIZER_STATE=""`.
- `LEARNING_RATE=2e-5` (default V13 produccion; antes V12 usaba `5e-5`): learning rate del
  fine-tune. 1e-5 (default original) no movia el modelo (~0.7% de deriva por
  bloque de 500 pasos); 5e-5 resulto ser mas riesgo del necesario (ver
  [configuracion-v13-entrenamiento/02_EVALUACION_CAUSAS_CHATGPT.md](../configuracion-v13-entrenamiento/02_EVALUACION_CAUSAS_CHATGPT.md)); 2e-5 es la
  receta actual para solo IP-Adapter con batch 1 en este hardware.
- Correccion de la revision visual: `comparar_calidad_v9.py` se invoca con
  `PYTHONPATH=<repo>` para que importe `src.*` (antes fallaba siempre con
  `ModuleNotFoundError: No module named 'src'`).

### V13 (2026-08-30)

- `LEARNING_RATE` baja de `5e-5` (V12) a `2e-5`: el A/B controlado de V13 mostro
  que `5e-5` era mas riesgo del necesario sin mejora clara adicional (ver
  [configuracion-v13-entrenamiento/04_RESULTADO_AB.md](../configuracion-v13-entrenamiento/04_RESULTADO_AB.md)).

### V14 (2026-09-10)

- Reinicio DESDE CERO desde el modelo oficial `yisol/IDM-VTON` en vez de
  `result_train_night/checkpoint-250`: ese checkpoint heredado derivaba de
  `result_train/checkpoint-100` y degradaba la transferencia de la prenda (ver
  [configuracion-v13-entrenamiento/04_RESULTADO_AB.md](../configuracion-v13-entrenamiento/04_RESULTADO_AB.md)).
- `BASE_CHECKPOINT` y `COMPACT_CHECKPOINT_ROOT`/`AUTO_OUTPUT_ROOT` pasan a
  `result_train_v14/base_oficial` y `result_train_v14/produccion_continua`.
- `LOG_ROOT` pasa a `logs/produccion_continua_v14` (el log de cron tambien se
  actualizo en `instalar_cron_watchdog.sh`).
- Detalle completo del estado y progreso acumulado en
  [configuracion-v14-entrenamiento/10_PLAN_IMPLEMENTACION_V14.md](../configuracion-v14-entrenamiento/10_PLAN_IMPLEMENTACION_V14.md).

## Donde quedan los artefactos

- Log principal watchdog:
  - `/home/uceda/Documents/IDM-VTON/logs/produccion_continua_v14/watchdog.log`
- Log de cron:
  - `/home/uceda/Documents/IDM-VTON/logs/produccion_continua_v14/cron_watchdog.log`
- Logs por corrida:
  - `/home/uceda/Documents/IDM-VTON/logs/produccion_continua_v14/train_run_YYYYmmdd_HHMMSS.log`
- Metadatos por corrida (inicio y fin, con exit_code):
  - `/home/uceda/Documents/IDM-VTON/logs/produccion_continua_v14/train_run_YYYYmmdd_HHMMSS.meta`
- Salida de modelos (checkpoints compactos, cadena de produccion real):
  - `/home/uceda/Documents/IDM-VTON/result_train_v14/produccion_continua/run_YYYYmmdd_HHMMSS/`

## Validacion rapida

```bash
# 1) Verificar cron
crontab -l

# 2) Ver actividad watchdog
tail -n 50 /home/uceda/Documents/IDM-VTON/logs/produccion_continua_v14/watchdog.log

# 3) Ver si hay entrenamiento activo
ps -eo pid,ppid,cmd --forest | grep -E 'train_xl.py|accelerate launch' | grep -v grep

# 4) Ver ultimos checkpoints compactos de la cadena real
find /home/uceda/Documents/IDM-VTON/result_train_v14/produccion_continua -maxdepth 3 -type d -name 'checkpoint-*' | sort -V
```

## Desinstalar tarea programada

```bash
cd "/home/uceda/Documents/IDM-VTON/entrenamiento continuo"
./desinstalar_cron_watchdog.sh
crontab -l
```

## Notas importantes sobre calidad del modelo

- Si, en general mas steps pueden mejorar el modelo, pero no siempre linealmente.
- Entrenar en bloques de 500 steps puede mejorar mientras la perdida y resultados visuales sigan mejorando.
- Tambien existe riesgo de sobreajuste (overfitting) si se repite demasiado sobre el mismo dataset.
- Recomendacion: validar visualmente cada cierto numero de checkpoints (por ejemplo cada 500 o 1000 steps, usando [configuracion-v9-entrenamiento/comparar_calidad_v9.py](../configuracion-v9-entrenamiento/comparar_calidad_v9.py), que ya corre rapido con la correccion de dtype) y detener/pausar (`PAUSAR_WATCHDOG`) cuando ya no haya mejora clara o se quiera decidir si promover un checkpoint.
- Nota sobre limpieza automatica: desde el 2026-08-23 el watchdog borra los checkpoints intermedios de cada bloque de 500 (deja solo el final) para no llenar el disco. Esto significa que ya no se puede comparar, por ejemplo, el checkpoint-100 contra el checkpoint-400 de un mismo bloque una vez que ese bloque termino; solo quedan los checkpoints "de cierre" de cada bloque (cada 500 pasos). Si se necesita mas granularidad para una comparacion puntual, pausar el watchdog (`PAUSAR_WATCHDOG`) antes de que termine el bloque de interes y copiar manualmente el checkpoint intermedio deseado a otra carpeta.

## Ver tambien

- [../README.md](../README.md) — comandos rapidos de arranque/parada de la app y del entrenamiento.
- [../GUIA_ENTRENAMIENTO_IDMVTON.md](../GUIA_ENTRENAMIENTO_IDMVTON.md) — version manual (paso a paso) de este mismo entrenamiento.
- [../configuracion-v14-entrenamiento/README.md](../configuracion-v14-entrenamiento/README.md) — configuracion y receta activas actualmente.
- [../INDEX.md](../INDEX.md) — indice completo de toda la documentacion del proyecto.
