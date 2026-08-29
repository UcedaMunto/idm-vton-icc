# Version 8 (propuesta): paralelismo de CPU sin cambiar el modelo

Fecha de creacion: 2026-08-22

## Estado

`PROPUESTA_NO_IMPLEMENTADA`. Esta carpeta contiene solo analisis y diseno, tomando como base `../configuracion-v7-entrenamiento` (V7 Tier 1 ya implementado y probado). No se ha modificado `train_xl.py` para nada de lo aqui descrito, no se ha instalado nada en cron y no se ha ejecutado ningun entrenamiento con estos cambios.

## Hallazgo que origina esta propuesta

Durante la comparacion de calidad de V7 (`comparar_calidad_v7.py`, corrida real en curso al momento de este analisis), se observo:

- La maquina tiene 12 nucleos/hilos logicos (`nproc`).
- El proceso Python tiene 13 hilos del sistema operativo (`ps -T`), consistente con que PyTorch si crea hilos.
- `torch.get_num_threads()` por defecto es 6 en este entorno (no 1, no 12).
- No hay ninguna variable de entorno `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` fijada en el proceso.
- Sin embargo, la observacion de uso de CPU por nucleo mostro un patron consistente con un nucleo saturado y el resto casi inactivos durante gran parte del computo (`ps` reporta ~100-104% de CPU total para el proceso principal, es decir, aproximadamente 1 nucleo de 12, no varios).

Esto es una discrepancia real: PyTorch esta configurado para usar hasta 6 hilos de paralelismo intra-operacion, pero el trabajo observado se comporta como si solo aprovechara uno. La causa mas probable, coherente con la arquitectura de este proyecto, no es una configuracion faltante sino la naturaleza del computo:

1. El UNet (y su version modificada con IP-Adapter/GarmentNet) ejecuta muchas capas pequenas de forma secuencial en Python (bucles de atencion, hooks personalizados, uniones de tensores) con `batch_size=1`. Cada operacion individual (conv/matmul) es demasiado pequena para que el paralelismo intra-operacion de PyTorch (basado en particionar una sola operacion grande entre hilos) aporte beneficio; el tiempo total termina dominado por la sobrecarga secuencial de Python y el dispatcher de PyTorch, que es inherentemente de un solo hilo (GIL).
2. El script de comparacion anade, ademas, hooks propios (`register_forward_pre_hook`/`register_forward_hook`) que mueven tensores entre CPU y GPU en cada llamada a GarmentNet; esa sincronizacion (`.to(device)`, `.item()` en otros puntos del pipeline) crea puntos de espera que tampoco se benefician de mas hilos.

Conclusion: aumentar el numero de hilos de PyTorch no es garantia de mejora, y debe probarse empiricamente, no asumirse. Esta carpeta documenta un conjunto de mejoras candidatas, ordenadas por riesgo, para intentar aprovechar mejor los 12 nucleos disponibles sin cambiar ningun resultado numerico del entrenamiento.

## Documentos

1. [01_PROPUESTA_PARALELISMO_CPU.md](01_PROPUESTA_PARALELISMO_CPU.md): diagnostico detallado y catalogo de mejoras candidatas (hilos de PyTorch, variables de entorno de BLAS, paralelismo a nivel de proceso, dataloader workers, batching).
2. [02_PLAN_VALIDACION_V8.md](02_PLAN_VALIDACION_V8.md): como probar cada mejora de forma aislada, criterios de aceptacion y de retroceso, y como se combina con V7 (GPU) sin mezclar variables en el mismo experimento.

## Principios heredados de V6/V7 (no se relajan aqui)

- Ninguna mejora se activa por defecto ni se instala en cron sin medicion previa.
- Un cambio a la vez: nunca combinar un cambio de hilos con un cambio de resolucion, batch, optimizador o Tier de V7 en el mismo experimento.
- Toda mejora que pueda alterar el orden de reduccion en operaciones de punto flotante (paralelismo intra-operacion, orden de acumulacion) se trata como cambio potencialmente numerico (`N`) y requiere el mismo protocolo de comparacion de calidad ya usado en V6/V7, no solo una medicion de velocidad.
- Ninguna mejora aqui debe degradar la fiabilidad ya lograda en V7 (ausencia de thrashing de RAM); cualquier cambio que aumente el paralelismo y por tanto el uso simultaneo de memoria debe revalidarse contra el techo de RAM ya documentado.
