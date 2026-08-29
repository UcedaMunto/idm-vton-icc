# Version 9: logging de progreso por paso + pruebas progresivas 5/10/15 steps

Fecha de creacion: 2026-08-22

## Estado

`CAUSA_RAIZ_ENCONTRADA_Y_CORREGIDA`. Se salta de V7 a V9 porque V8 (ajuste de hilos de CPU) se probo y se descarto como palanca de velocidad (ver `../configuracion-v8-entrenamiento/01_PROPUESTA_PARALELISMO_CPU.md`, seccion "Resultado real"). El problema real no era de hilos ni de visibilidad: era que **GarmentNet se ejecutaba en `bfloat16` sobre CPU, un dtype sin kernels optimizados en esta version de PyTorch**, causando que un solo forward tardara mas de 20 minutos sin completar. Corregido a `float32` (el dtype que el proyecto original siempre uso para esta ruta, aunque el original la ejecutaba en GPU). Ver [01_CAUSA_RAIZ_Y_CORRECCION_DTYPE.md](01_CAUSA_RAIZ_Y_CORRECCION_DTYPE.md) para el analisis completo, la medicion aislada y el resultado real verificado.

## Que hace V9

1. `comparar_calidad_v9.py`: copia de `../configuracion-v7-entrenamiento/comparar_calidad_v7.py` con logging explicito en puntos clave:
   - Carga de modelos (tiempo total).
   - Colocacion de dispositivos.
   - Cada llamada a GarmentNet (CPU): numero de llamada y duracion. Originalmente en `bfloat16` (causo la investigacion de la causa raiz), corregido a `float32`.
   - **Cada paso de denoising** (via `callback_on_step_end` del pipeline): numero de paso, tiempo transcurrido desde el inicio del bucle, y RSS maxima del proceso en ese momento.
   - Guardado de la imagen final.
2. `ejecutar_pruebas_progresivas_v9.sh`: ejecuta, de forma secuencial (nunca en paralelo), tres corridas con `--num_inference_steps` en 5, 10 y 15, cada una con su propio log en `logs/` y su propia carpeta de salida en `comparacion_progresiva/`. Aborta la secuencia si detecta otro proceso de entrenamiento/inferencia activo o si una corrida falla.

Ninguna corrida se instala en cron; todas son manuales y supervisadas, igual que en V6/V7/V8.

## Por que 5, 10 y 15 pasos progresivos

Antes de decidir cuantos pasos usar para las comparaciones de calidad de V6/V7 (que hasta ahora usaban 30 por defecto, heredado de `inference.py`), conviene medir el tiempo real por paso con este hardware y confirmar visualmente a partir de que numero de pasos la imagen deja de mejorar de forma perceptible. Tres corridas cortas y crecientes, con logging por paso, permiten:

- Ver el tiempo por paso de GarmentNet en CPU de forma directa (en vez de inferirlo de la duracion total).
- Detectar temprano cualquier problema (por ejemplo, si el paso 1 ya tarda mas de lo esperado, no hace falta esperar a que termine toda la corrida para saberlo).
- Tener, al final, tres imagenes generadas con el mismo par y semilla pero distinto numero de pasos, para decidir el numero minimo de pasos aceptable para las comparaciones futuras.

## Que NO hace V9

- No cambia `train_xl.py` ni ningun parametro de entrenamiento.
- No cambia el numero de pasos de inferencia usado por V6/V7 como "oficial"; eso se decide despues de ver los resultados de esta carpeta.
- No reemplaza el protocolo completo de comparacion de calidad de `../configuracion-v6-entrenamiento/04_PLAN_COMPARACION_CALIDAD.md` (rubrica ciega, conjunto de 30+ pares); esto es una herramienta de diagnostico y ajuste rapido, no la comparacion final.

## Como ejecutar (referencia)

```bash
cd /home/uceda/Documents/IDM-VTON
./configuracion-v9-entrenamiento/ejecutar_pruebas_progresivas_v9.sh
```

Seguimiento en vivo del progreso de la corrida actual (en otra terminal):

```bash
tail -f /home/uceda/Documents/IDM-VTON/configuracion-v9-entrenamiento/logs/steps5_*.log
```

## Documentos

1. [01_CAUSA_RAIZ_Y_CORRECCION_DTYPE.md](01_CAUSA_RAIZ_Y_CORRECCION_DTYPE.md): comparacion contra el proyecto original, microbenchmark aislado `float32` vs `bfloat16` en CPU, correccion aplicada al script de comparacion, resultado real verificado, y pendientes para decidir si se aplica el mismo cambio a `train_xl.py`.
2. [02_RESULTADOS_PRUEBAS_PROGRESIVAS.md](02_RESULTADOS_PRUEBAS_PROGRESIVAS.md): tabla comparativa de tiempo por corrida (5/10/15 pasos) y verificacion de uso de GPU en paralelo durante las corridas, tras aplicar la correccion.
3. [03_APLICACION_FIX_A_V7.md](03_APLICACION_FIX_A_V7.md): hallazgo de que las configuraciones `.env` de V4/V5/V6/V7 sobrescribian `--garmentnet_dtype` a `bfloat16` para el entrenamiento real (aunque `train_xl.py` ya traia `float32` como default), evidencia del impacto (~14x mas lento por paso), alcance real (solo smoke tests de 1 paso, sin perdida de entrenamiento prolongado), correccion aplicada a los 4 archivos `.env`, y verificacion con una nueva corrida de smoke test V7.
