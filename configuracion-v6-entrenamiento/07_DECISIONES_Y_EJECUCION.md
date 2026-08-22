# Decisiones adoptadas y acciones ejecutadas

Fecha de ejecucion: 2026-08-22

Este documento registra que las 4 recomendaciones se adoptaron como decision y que acciones concretas se ejecutaron sobre el proyecto en vivo. No sustituye a los documentos 01-06; los complementa con el estado real posterior a la intervencion.

## Decision adoptada

1. No reiniciar el entrenamiento desde cero. Los checkpoints compactos existentes en `result_train_v4` se conservan como linea base valida para V6.
2. Pausar la automatizacion (cron/watchdog) hasta cerrar las brechas reales de linaje/checksum descritas en la Fase 1 corregida del plan V6, aunque el supervisor de fallos ya es funcional (ver erratum en `01_AUDITORIA_CAMBIOS_Y_FALLOS.md`).
3. No sumar mas updates "a ciegas": antes de reanudar entrenamiento continuo, ejecutar el protocolo de comparacion A/B de `04_PLAN_COMPARACION_CALIDAD.md` contra `checkpoint-250` y contra el propio historial V4.
4. Tratar `result_train_v4/v4_run_20260822_101401/checkpoint-1` como el ultimo estado valido y punto de partida de V6; no se descarta ni se sobrescribe.

## Acciones ejecutadas en este turno

### 1. Verificacion de estado antes de actuar

Se confirmo, antes de detener nada:

- Proceso activo: `accelerate launch train_xl.py` (PID 41981, hijos 41999/42400), corrida `v4_run_20260822_104901`, reanudada desde `result_train_v4/v4_run_20260822_101401/checkpoint-1`.
- Progreso de esa corrida: `Steps: 0%|...| 0/1` -- no habia completado ningun update ni generado checkpoint propio.
- Cron instalado y activo cada minuto.
- Sin archivos de pausa presentes en ese momento.
- Ultimo checkpoint valido preexistente: `result_train_v4/v4_run_20260822_101401/checkpoint-1/manifest.json`.

Conclusion de la verificacion: detener la corrida en ese punto no perdia ningun update completado.

### 2. Pausa de automatizacion antes de detener el proceso

```bash
touch "entrenamiento continuo/PAUSAR_WATCHDOG"
```

Se coloco el archivo de pausa manual primero, para que cron no relanzara una corrida nueva en el minuto siguiente al detener el proceso.

### 3. Detencion ordenada del entrenamiento

```bash
pkill -TERM -f 'accelerate launch train_xl.py'
pkill -TERM -f 'python3.10 train_xl.py'
```

Resultado: los procesos terminaron con `SIGTERM` (sin necesidad de `SIGKILL`). Verificacion posterior confirmo `sin_procesos_activos`.

### 4. Hallazgo durante la detencion: el supervisor real SI reacciono

Al enviar `SIGTERM`, el supervisor ya presente en `entrenamiento continuo/watchdog_entrenamiento.sh` (ver erratum en el documento 01) escribio automaticamente:

```text
entrenamiento continuo/PAUSAR_POR_ERROR
run_id=v4_run_20260822_104901
exit_code=143
expected_checkpoint=/home/uceda/Documents/IDM-VTON/result_train_v4/v4_run_20260822_104901/checkpoint-1
log_file=/home/uceda/Documents/IDM-VTON/logs/training_v4/v4_run_20260822_104901.log
failed_at=2026-08-22 10:58:06
```

Esto es el comportamiento esperado y correcto del supervisor ante una corrida detenida manualmente (exit code 143 = terminada por `SIGTERM`), no un fallo del proyecto. Se decide **no borrar este archivo todavia**: mientras exista, junto con `PAUSAR_WATCHDOG`, cron no puede relanzar ningun entrenamiento, lo cual coincide con la recomendacion 2. Borrar `PAUSAR_POR_ERROR` requiere una decision explicita posterior del usuario, conforme a la regla ya documentada en `entrenamiento continuo/README.md`.

### 5. Verificacion posterior de integridad

Confirmado tras la detencion:

- `result_train_v4/v4_run_20260822_104901/` quedo vacio (sin checkpoint parcial, sin `.tmp` residual).
- `result_train_v4/v4_run_20260822_101401/checkpoint-1/manifest.json` permanece intacto y es el ultimo estado valido.
- Cron sigue instalado (no se desinstalo), pero bloqueado por dos archivos de pausa independientes: `PAUSAR_WATCHDOG` (manual) y `PAUSAR_POR_ERROR` (automatico).

## Correccion registrada sobre el hallazgo F-01

Durante esta intervencion se detecto que la auditoria original describia el watchdog como no funcional (sin config, sin supervisor, sin disyuntor). Esa descripcion se baso en una lectura de archivo truncada/desactualizada. La prueba en vivo de esta seccion demuestra lo contrario: el supervisor existente funciona. El documento `01_AUDITORIA_CAMBIOS_Y_FALLOS.md` ya se corrigio con un erratum fechado y F-01 se reclasifico de `BLOQUEANTE` a `MEDIO`. Las brechas reales que persisten (validacion de linaje/checksum, formato de configuracion no ejecutable) siguen abiertas y programadas en la Fase 1 corregida de `03_PLAN_CORRECCIONES_V6.md`.

## Estado del proyecto al cierre de este turno

| Elemento | Estado |
|---|---|
| Entrenamiento | Detenido, sin procesos activos |
| Cron | Instalado, bloqueado por doble pausa |
| `PAUSAR_WATCHDOG` | Presente (pausa manual) |
| `PAUSAR_POR_ERROR` | Presente (disyuntor automatico por `SIGTERM` intencional) |
| Ultimo checkpoint valido | `result_train_v4/v4_run_20260822_101401/checkpoint-1` |
| Corrida interrumpida | `v4_run_20260822_104901`, vacia, sin perdida de progreso previo |
| Gradio | No se detecto activo durante la verificacion |

## Proximos pasos pendientes de decision del usuario

1. Revisar y decidir cuando levantar `PAUSAR_WATCHDOG` y `PAUSAR_POR_ERROR` (no se levantan automaticamente).
2. Ejecutar el protocolo A/B de `04_PLAN_COMPARACION_CALIDAD.md` usando `checkpoint-1` de `v4_run_20260822_101401` como candidato antes de decidir si conviene seguir esa linea o iniciar una nueva bajo un esquema V6 corregido.
3. Cerrar las brechas de la Fase 1 corregida (linaje/checksum, config no ejecutable) antes de reinstalar cualquier automatizacion continua.
4. No se implemento en este turno ningun cambio a `train_xl.py` (resolucion, dtype, clipping, `time_ids`, semilla global): esos siguen siendo cambios matematicos que requieren experimento aislado segun la Fase 4-5 del plan, y no se ejecutan bajo presion para evitar degradar el modelo.
