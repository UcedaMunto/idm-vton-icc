# V4: control de RAM y entrenamiento recuperable

## Motivo

V3 logro guardar checkpoints compactos, pero los bloques automaticos siguieron muriendo antes del primer checkpoint. La causa confirmada por `journalctl` fue OOM de RAM del sistema: procesos Python con aproximadamente 25-29 GiB residentes fueron terminados por el kernel. El cron relanzaba otro proceso cada minuto, creando un ciclo nocturno sin avance.

## Cambios implementados

### 1. GarmentNet congelado en BF16

`train_xl.py` acepta `--garmentnet_dtype` y V4 usa `bfloat16` para el UNet congelado de GarmentNet. Sus entradas se convierten al mismo dtype y sus features vuelven al dtype del UNet entrenable. Esto reduce la huella de RAM del modulo que no se entrena.

Si el CPU no soporta BF16 en alguna operacion, el log mostrara el error y se debe probar `--garmentnet_dtype=float32`; no se debe automatizar el fallback porque podria ocultar un perfil inestable.

### 2. No cargar optimizer por defecto

V3 guardaba `optimizer_state.pt` de aproximadamente 3.2 GiB. V4 reanuda pesos entrenables sin cargar ese estado. `--resume_optimizer_state` queda disponible solo como opt-in y no debe usarse en esta maquina hasta demostrar que hay RAM suficiente.

### 3. Disyuntor del watchdog

El archivo `entrenamiento continuo/PAUSAR_POR_ERROR` permite bloquear relanzamientos despues de un fallo. El watchdog V4 no debe ejecutarse sin revisar logs; un `SIGKILL`/OOM requiere diagnostico, no otro intento identico.

### 4. Prueba escalonada

La configuracion empieza en 1 update. Solo despues de que ese update termine con `checkpoint-1` y memoria estable se cambia `MAX_TRAIN_STEPS` a 10. La resolucion, batch y workers se mantienen sin cambios.

## Ejecucion

```bash
cd /home/uceda/Documents/IDM-VTON
chmod +x configuracion-v4-entrenamiento/ejecutar_prueba_v4.sh
bash -n train_xl.py configuracion-v4-entrenamiento/ejecutar_prueba_v4.sh
./configuracion-v4-entrenamiento/ejecutar_prueba_v4.sh
```

## Criterios de aprobacion

- `exit_code=0`.
- `Steps: 100%|...| 1/1`.
- Se crea un checkpoint compacto con `trainable_state.pt`, `optimizer_state.pt` y `manifest.json`.
- No hay OOM/SIGKILL en `journalctl`.
- RAM disponible durante el guardado permanece por encima de 4 GiB.
- No se reactiva cron todavia.

## Reanudar despues de aprobar V4

La automatizacion solo se puede reactivar despues de la prueba de 1 update y una prueba posterior de 10 updates. Antes de hacerlo, actualizar `config_v4.env`, revisar el log y colocar el disyuntor en estado limpio.
