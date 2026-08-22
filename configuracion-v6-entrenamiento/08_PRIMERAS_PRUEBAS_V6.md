# Primeras pruebas V6 (Fase 7)

Fecha: 2026-08-22

## Alcance de esta prueba

Se implemento y ejecuto unicamente la etapa 1 de la Fase 7 del plan (`03_PLAN_CORRECCIONES_V6.md`): validacion estatica sin entrenamiento. No se ejecuto GPU, no se lanzo `accelerate`/`train_xl.py`, no se creo ningun directorio de resultados y no se modificaron `PAUSAR_WATCHDOG` ni `PAUSAR_POR_ERROR`.

## Artefactos creados

- [config_v6.env](config_v6.env): primera revision de configuracion V6. `MODE=warm_start_weights` (no se declara continuidad exacta porque el checkpoint padre no incluye estado de optimizer ni RNG, ver F-03). `PARENT_CHECKPOINT_ID` referencia explicitamente `v4_run_20260822_101401/checkpoint-1`, el ultimo estado valido segun `07_DECISIONES_Y_EJECUCION.md`.
- [validar_v6.sh](validar_v6.sh): script de solo lectura que valida precondiciones y construye el comando efectivo, sin ejecutarlo.

Ninguno de los dos archivos esta instalado en cron ni se invoca automaticamente desde otro script del proyecto.

## Resultado de la ejecucion

```text
VALIDACION_ESTATICA_OK (no se lanzo ningun proceso de entrenamiento)
```

Checks superados:

- Config cargada y con hash registrado (`4c7f37bc83168c7385ba76a6870588e4e529bc003c7126630d7107c50f3a6ac9`).
- Sin procesos de entrenamiento ni Gradio activos.
- Checkpoint base (`checkpoint-250`) valido.
- Checkpoint de reanudacion valido, con manifest legible: `base_checkpoint=checkpoint-250`, `completed_optimizer_updates=1`, `checkpoint_type=compact_training_state`, `width=448 height=576`, `train_ip_adapter_only=True`, `garmentnet_dtype=bfloat16`. Esto confirma linaje coherente con lo esperado antes de construir el comando.
- Disco: 161 GiB libres (minimo 90). RAM: 28 GiB disponibles (minimo 4).
- Resolucion multiplo de 8; `checkpointing_steps <= max_train_steps`.
- Directorio de salida propuesto (`result_train_v6/smoke_v6_<timestamp>`) no existe todavia; no se creo.

Advertencias esperadas y no bloqueantes:

- `PAUSAR_WATCHDOG` y `PAUSAR_POR_ERROR` siguen presentes (correcto; automatizacion sigue bloqueada).
- `MODE=warm_start_weights` se marca explicitamente como no-continuidad-exacta.
- La tabla de parametros solicitados vs efectivos vuelve a mostrar que `mixed_precision`, `optimizer` y `xformers` solicitados no coinciden con los efectivos bajo `low_vram_training` (ver `02_MATRIZ_PARIDAD_Y_PARAMETROS.md`); esto es un recordatorio, no un fallo de esta revision.

## Lo que esta prueba demuestra

1. El checkpoint `v4_run_20260822_101401/checkpoint-1` es utilizable como punto de partida verificado para V6 (manifest legible, hashes calculables, linaje declarado coherente).
2. La configuracion V6 puede validarse de forma completamente segura, sin GPU y sin posibilidad de arrancar un entrenamiento por accidente.
3. El comando que se ejecutaria quedo impreso para inspeccion manual antes de cualquier ejecucion real.

## Lo que esta prueba NO hace

- No ejecuta ningun update real ni toca la GPU.
- No prueba reanudacion continua vs interrumpida (Fase 2/Experimento B).
- No valida los defectos matematicos heredados (`time_ids`, clipping, semilla global).
- No decide si el candidato V6 iguala o mejora la calidad frente a `checkpoint-250` (Fase 5/Experimento C).

## Siguiente paso propuesto (requiere confirmacion explicita antes de ejecutarlo)

Ejecutar un unico update real de humo (Fase 7, paso 3) con `config_v6.env` tal cual quedo validado:

1. Levantar `PAUSAR_WATCHDOG` no es necesario para esta prueba manual (el lanzamiento seria manual, no via cron).
2. `PAUSAR_POR_ERROR` puede permanecer; esta prueba no depende del watchdog automatico.
3. Ejecutar exactamente el comando impreso en la seccion 8 de `validar_v6.sh`, de forma manual y supervisada, confirmando antes que no hay Gradio activo.
4. Verificar al finalizar: `exit_code=0`, existencia de `result_train_v6/<run>/checkpoint-1/manifest.json` y `trainable_state.pt`, ausencia de OOM/SIGKILL, RAM disponible >4 GiB durante el guardado.
5. Solo despues de aprobar este update se plantea la prueba de reanudacion 1+1 (Experimento B) y, mas adelante, la comparacion de calidad (Experimento C) contra `checkpoint-250`.

Este paso no se ejecuto todavia porque implica uso real de GPU/tiempo y decidir si se levantan temporalmente las pausas; se deja pendiente de confirmacion explicita del usuario antes de lanzarlo.
