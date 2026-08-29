# V5: automatizacion segura del entrenamiento continuo

## Alcance

V5 documenta la automatizacion incorporada despues de preparar el perfil V4. No
cambia el modelo ni los parametros de memoria aprobados por V4: organiza su
ejecucion continua mediante cron, evita corridas duplicadas y detiene los
relanzamientos cuando una corrida no publica un checkpoint valido.

El entrenamiento activo al crear esta documentacion sigue usando
`configuracion-v4-entrenamiento/config_v4.env` y escribe en `result_train_v4`.
Esta carpeta conserva una referencia reproducible de ese estado; no sustituye la
configuracion activa automaticamente.

## Cambios documentados

### 1. Watchdog conectado a V4

`entrenamiento continuo/watchdog_entrenamiento.sh` carga un unico archivo de
configuracion y ejecuta bloques recuperables. La configuracion activa es V4; el
archivo `config_v5.env` de esta carpeta permite migrar explicitamente cuando se
decida abrir una nueva linea de resultados V5.

### 2. Una sola corrida

El watchdog usa `flock` para impedir dos evaluaciones simultaneas y comprueba
`train_xl.py` antes de lanzar otra corrida. Cron puede llamarlo cada minuto sin
duplicar un entrenamiento que siga activo.

### 3. Cadena de checkpoints

La seleccion sigue este orden:

1. Ultimo `trainable_state.pt` de la linea de entrenamiento configurada.
2. Ultimo checkpoint compacto del directorio de fallback.
3. Validacion obligatoria de `manifest.json` antes de reanudar.

Para la linea activa, `result_train_v4` es el destino y `result_train_v3` es el
fallback de arranque. Cada corrida crea un directorio independiente y nunca
sobrescribe resultados anteriores.

### 4. Perfil de memoria conservador

- GarmentNet congelado usa `bfloat16`.
- Se reanudan pesos entrenables sin cargar el estado del optimizer.
- El perfil low-VRAM mantiene resolucion 448x576, batch 1 y un worker.
- `LD_LIBRARY_PATH` incluye las bibliotecas del entorno Conda.
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64` limita fragmentacion CUDA.
- El entrenamiento empieza en bloques de un update con checkpoint cada update.

### 5. Supervisor y disyuntor

El watchdog deja un supervisor esperando el resultado de `accelerate launch`.
Una corrida solo se considera correcta cuando termina con codigo cero y publica
`manifest.json` y `trainable_state.pt` en el checkpoint esperado.

Ante un error, `SIGKILL` o salida incompleta, crea:

```text
/home/uceda/Documents/IDM-VTON/entrenamiento continuo/PAUSAR_POR_ERROR
```

Mientras exista ese archivo, cron no vuelve a entrenar. Debe revisarse el log y
corregirse la causa antes de borrarlo.

### 6. Trazabilidad

Cada corrida registra un archivo `.meta` con identificador, PID del supervisor,
checkpoint base, checkpoint de reanudacion, salida, log, steps y espacio libre.
Los logs del watchdog, cron y entrenamiento se concentran en el `LOG_ROOT` de la
configuracion.

## Estado al crear V5

Fecha: 2026-08-22.

- Cron instalado cada minuto, con una sola entrada.
- Corrida activa: `v4_run_20260822_093939`.
- Base: `result_train_night/checkpoint-250`.
- Reanudacion: `result_train_v3/smoke10_20260820_223032/checkpoint-10`.
- Objetivo: un update y `checkpoint-1`.
- Disyuntor por error: inactivo.
- Todavia no existia un checkpoint V4 publicado al capturar el estado.

## Crontab instalada

```cron
* * * * * /bin/bash "/home/uceda/Documents/IDM-VTON/entrenamiento continuo/watchdog_entrenamiento.sh" >> "/home/uceda/Documents/IDM-VTON/logs/training_v4/cron_watchdog.log" 2>&1
```

## Operacion

Consultar el estado sin modificar procesos:

```bash
cd /home/uceda/Documents/IDM-VTON
./configuracion-v5-entrenamiento/monitorear_estado_v5.sh
```

Monitor continuo:

```bash
watch -n 5 /home/uceda/Documents/IDM-VTON/configuracion-v5-entrenamiento/monitorear_estado_v5.sh
```

En low-VRAM, una GPU con poca utilizacion no implica por si sola un bloqueo: gran
parte del forward se ejecuta en CPU. Un proceso `train_xl.py` con CPU acumulada y
estado `R` sigue trabajando aunque la barra permanezca en 0% hasta completar el
primer update. Mas de 8 GiB de swap indica presion de memoria y puede prolongar
considerablemente ese update.

Pausa manual antes de usar Gradio:

```bash
touch "/home/uceda/Documents/IDM-VTON/entrenamiento continuo/PAUSAR_WATCHDOG"
```

Reanudar despues de comprobar que no existe Gradio ni otro entrenamiento:

```bash
rm -f "/home/uceda/Documents/IDM-VTON/entrenamiento continuo/PAUSAR_WATCHDOG"
/bin/bash "/home/uceda/Documents/IDM-VTON/entrenamiento continuo/watchdog_entrenamiento.sh"
```

No borrar `PAUSAR_POR_ERROR` sin inspeccionar primero el log indicado dentro del
archivo.

## Migracion opcional a una linea V5

`config_v5.env` apunta a `result_train_v5` y `logs/training_v5`, con fallback a
`result_train_v4`. Para activarlo de forma controlada se debe esperar a que no
haya entrenamiento, validar un checkpoint V4 y cambiar `CONFIG_FILE` en el
watchdog o en su entorno de ejecucion. La crontab actual no realiza esa migracion.

## Criterios para aumentar el bloque

Mantener `MAX_TRAIN_STEPS=1` hasta comprobar:

- `Steps: 100%|...| 1/1`.
- Existencia de `checkpoint-1/manifest.json` y `trainable_state.pt`.
- Ausencia de OOM, `SIGKILL`, NaN o errores de dtype.
- RAM disponible superior a 4 GiB durante guardado.
- Disyuntor ausente tras finalizar.

Solo despues se prueba un bloque de 10 updates, manteniendo checkpoint frecuente
y observacion de RAM, swap y logs.