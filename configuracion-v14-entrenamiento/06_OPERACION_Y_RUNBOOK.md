# 06 - Operacion y runbook

Fecha: 2026-09-09. Comandos listos para copiar. Rutas absolutas.

## 6.1 Ver el estado

```bash
# Cron instalado
crontab -l

# Ultimas decisiones del watchdog
tail -n 40 "/home/uceda/Documents/IDM-VTON/logs/produccion_continua_v14/watchdog.log"

# Hay entrenamiento corriendo?
pgrep -af 'train_xl.py'

# GPU
nvidia-smi

# Espacio y RAM
df -h /home/uceda/Documents ; free -h

# Ultimos checkpoints de la cadena
find /home/uceda/Documents/IDM-VTON/result_train_v14/produccion_continua \
  -maxdepth 2 -type d -name 'checkpoint-*' | sort -V
```

## 6.2 Pausar y reanudar el entrenamiento

```bash
# PAUSAR (no lanza bloques nuevos; el bloque en curso termina)
touch "/home/uceda/Documents/IDM-VTON/entrenamiento continuo/PAUSAR_WATCHDOG"

# REANUDAR
rm -f "/home/uceda/Documents/IDM-VTON/entrenamiento continuo/PAUSAR_WATCHDOG"
# y, para no esperar al cron (hasta 1 min):
/bin/bash "/home/uceda/Documents/IDM-VTON/entrenamiento continuo/watchdog_entrenamiento.sh"

# Si el watchdog se autopauso por error, revisar y luego:
cat  "/home/uceda/Documents/IDM-VTON/entrenamiento continuo/PAUSAR_POR_ERROR"
rm -f "/home/uceda/Documents/IDM-VTON/entrenamiento continuo/PAUSAR_POR_ERROR"
```

Para un bloque manual con otros parametros (sin tocar el cron):

```bash
cd "/home/uceda/Documents/IDM-VTON/entrenamiento continuo"
MAX_TRAIN_STEPS=100 CHECKPOINTING_STEPS=50 LEARNING_RATE=2e-5 bash watchdog_entrenamiento.sh
```

Parametros mas utiles del watchdog (todos por variable de entorno):
`MAX_TRAIN_STEPS`, `CHECKPOINTING_STEPS`, `LEARNING_RATE`,
`RESUME_OPTIMIZER_STATE`, `WIDTH`, `HEIGHT`, `TRAIN_BATCH_SIZE`,
`BASE_CHECKPOINT`, `COMPACT_CHECKPOINT_ROOT`, `LOG_ROOT`, `MIN_FREE_GIB`.

## 6.3 Monitorear el bloque en curso

```bash
tail -f "$(ls -t /home/uceda/Documents/IDM-VTON/logs/produccion_continua_v14/train_run_*.log | head -1)"
cat    "$(ls -t /home/uceda/Documents/IDM-VTON/logs/produccion_continua_v14/train_run_*.meta | head -1)"
```

## 6.4 Exportar un checkpoint a un pipeline cargable por la app

> Ojo: `configuracion-v12-entrenamiento/exportar_prueba.sh` todavia tiene
> `BASE_CHECKPOINT` por defecto en la base V10 (`result_train_night/checkpoint-250`)
> y escribe en `result_train_v10/demos/`. Para V14 usar el script python directo:

```bash
CKPT=/home/uceda/Documents/IDM-VTON/result_train_v14/produccion_continua/run_20260909_100801/checkpoint-500
OUT=/home/uceda/Documents/IDM-VTON/result_train_v14/demos/v14_cum1500

PYTHONPATH=/home/uceda/Documents/IDM-VTON \
/home/uceda/miniconda3/envs/idm/bin/python \
  /home/uceda/Documents/IDM-VTON/exportar_checkpoint_para_demo.py \
  --base_checkpoint /home/uceda/Documents/IDM-VTON/result_train_v14/base_oficial \
  --compact_checkpoint "$CKPT" \
  --output_dir "$OUT"
```

El export tarda ~10 min y consume RAM. `OUT` no debe existir antes.

## 6.5 Cambiar el modelo que sirve la app

```bash
cd /home/uceda/Documents/IDM-VTON
./switch_model_version.sh custom /home/uceda/Documents/IDM-VTON/result_train_v14/demos/v14_cum1500
# volver al oficial:
./switch_model_version.sh current
./switch_model_version.sh status

# reiniciar la app para aplicar
pkill -f 'python .*gradio_demo/app.py' || true
nohup /home/uceda/Documents/IDM-VTON/run_gpu_with_logs.sh \
  > /home/uceda/Documents/IDM-VTON/logs/launcher_nohup.out 2>&1 &
curl -I -s http://127.0.0.1:7860 | head -n 1
```

Recordar: con entrenamiento en marcha **no** se puede levantar la app (12 GB).

## 6.6 Gestion de disco (retencion)

El disco es el limitante del entrenamiento continuo (doc 05). Politica segura:

```bash
# 1) En produccion solo hace falta el ultimo run_* para encadenar.
cd /home/uceda/Documents/IDM-VTON/result_train_v14/produccion_continua
ls -td run_* | tail -n +2          # LISTA candidatos (no borra nada)
ls -td run_* | tail -n +2 | xargs -r rm -rf   # BORRA (solo si el ultimo bloque termino OK)
df -h /home/uceda/Documents

# 2) Podar checkpoints preservados dejando solo los 2 ultimos runs.
cd /home/uceda/Documents/IDM-VTON/result_train_v14/pruebas_checkpoints
ls -td run_* | tail -n +3 | xargs -r rm -rf
```

**Nunca borrar:** `result_train_v14/base_oficial` (es un symlink al oficial),
el `run_*` mas reciente de `produccion_continua`, ni las carpetas `demos/` que
use la app. Antes de borrar, confirmar que no hay bloque activo (`pgrep -af train_xl.py`).

## 6.7 Respaldos

```bash
cd /home/uceda/Documents/IDM-VTON
# Configuracion y documentos (ligero)
tar czf "backups/v14_docs_$(date '+%Y%m%d').tgz" \
  configuracion-v14-entrenamiento \
  "entrenamiento continuo/watchdog_entrenamiento.sh" \
  "entrenamiento continuo/preservar_checkpoints_pruebas.sh" 2>/dev/null || true

# El artefacto que NO se puede regenerar: trainable_state.pt del mejor checkpoint.
cp -a <mejor_checkpoint>/trainable_state.pt backups/trainable_v14_mejor.pt
```

## 6.8 Problemas tipicos

| Sintoma | Causa probable | Accion |
|---|---|---|
| El watchdog no lanza ("espacio insuficiente") | `MIN_FREE_GIB=40` no se cumple | Purgar `run_*` (6.6) |
| El watchdog no lanza ("RAM insuficiente") | Menos de 3 GiB disponibles | Cerrar app/navegador |
| Aparece `PAUSAR_POR_ERROR` | Una corrida fallo o no genero checkpoint | Revisar `.meta`/`.log`, corregir, borrar el archivo |
| La app no levanta | Entrenamiento ocupando la GPU | Pausar (`PAUSAR_WATCHDOG`) |
| El export falla | `output_dir` ya existe | Elegir otro nombre |
| Nada arranca y `PAUSAR_WATCHDOG` existe | Pausa manual olvidada | `rm -f` del archivo |
| `ModuleNotFoundError: src` en revision | `PYTHONPATH` ausente | Ya corregido en el watchdog (V12) |

