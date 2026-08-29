# Version 7: uso de GPU sin exceder 12 GiB

Fecha de creacion: 2026-08-22

## Estado

`TIER1_IMPLEMENTADO_Y_PROBADO`. Tier 1 (modelos pequenos en GPU) se implemento en `train_xl.py` (flag `--hybrid_small_models_gpu`), junto con la correccion de RAM de V6. Se ejecutaron con exito un smoke test (1 update) y una prueba de reanudacion 1+1, ambos verificados manualmente (hashes, manifest, memoria). Una comparacion de calidad frente al perfil base esta en curso. Ningun cron fue instalado ni activado; toda ejecucion hasta ahora fue manual y supervisada. Ver [04_IMPLEMENTACION_Y_PRUEBAS.md](04_IMPLEMENTACION_Y_PRUEBAS.md) para el detalle completo, incluyendo los errores encontrados y corregidos durante el desarrollo del script de comparacion.

Nace directamente de la V6 (`../configuracion-v6-entrenamiento`) y de la medicion real hecha durante su primer smoke test: con `--low_vram_training`, la GPU se usaba a 348 MiB / 29% de utilizacion instantanea / 24.67 W (practicamente inactiva) mientras un solo update tardaba mas de 15 minutos, porque el UNet entrenable, GarmentNet, el VAE y los encoders de texto/imagen corrian enteramente en CPU.

## Pregunta que responde esta propuesta

¿Es posible mover mas computo a la GPU de 12 GiB sin superar su limite y sin cambiar la aritmetica del entrenamiento (mismos parametros entrenables, misma resolucion, mismo optimizador, mismo checkpoint) para reducir el tiempo por update?

Respuesta corta: si, mediante una estrategia de **residencia progresiva y secuencial**, no moviendo todo a la GPU de forma simultanea. Los documentos de esta carpeta explican por que la residencia simultanea de ambos UNets (el entrenable y GarmentNet) es incompatible con 12 GiB en fp32/fp16 combinados con el estado del optimizador, y proponen un diseno por etapas que se valida empiricamente antes de avanzar a la siguiente. Tier 1 ya esta implementado; Tier 2/3 siguen siendo propuesta.

## Documentos

1. [01_PROPUESTA_HIBRIDO_GPU_CPU.md](01_PROPUESTA_HIBRIDO_GPU_CPU.md): diseno tecnico por niveles (tiers), que se mueve a GPU en cada uno y por que.
2. [02_PRESUPUESTO_VRAM_Y_MEDICION.md](02_PRESUPUESTO_VRAM_Y_MEDICION.md): calculo de memoria por componente, cifras medidas (incluye conteo real de parametros de GarmentNet) vs. cifras publicas de referencia.
3. [03_PLAN_VALIDACION_V7.md](03_PLAN_VALIDACION_V7.md): orden de implementacion, criterios de aceptacion/retroceso y resultados reales de Tier 1.
4. [04_IMPLEMENTACION_Y_PRUEBAS.md](04_IMPLEMENTACION_Y_PRUEBAS.md): registro de los cambios de codigo aplicados, las dos pruebas de entrenamiento ejecutadas (smoke test y reanudacion 1+1) y el desarrollo del script de comparacion de calidad, con los errores encontrados y corregidos.

## Artefactos de codigo

- `train_xl.py`: modificado (correccion de RAM + flag `--hybrid_small_models_gpu`). Ver diff con `git diff -- train_xl.py`.
- `config_v7.env`, `validar_v7.sh`, `ejecutar_smoke_v7.sh`: perfil V7 Tier 1, validacion estatica y ejecucion supervisada (ninguno instalado en cron).
- `comparar_calidad_v7.py`: script de comparacion de calidad (base vs checkpoint compacto superpuesto), con manejo manual de dispositivos para caber en 12 GiB durante inferencia.



## Principios heredados de V6 (no se relajan aqui)

- Un cambio a la vez; nunca mezclar una mejora de velocidad con un cambio de resolucion, optimizador o alcance entrenable.
- Todo cambio de ubicacion CPU/GPU que tambien cambie el dtype (fp32 a fp16/bf16) se clasifica como cambio `N` (numerico) y requiere su propia comparacion, no solo una medicion de velocidad.
- Ninguna etapa se activa en cron ni se declara "lista" sin pasar primero por validacion estatica, luego un update real, luego una comparacion de calidad frente al candidato V6 ya validado.
- Si una etapa se acerca al techo seguro de VRAM (10.5 GiB de pico, mismo margen ya usado en los planes V3/V6), se retrocede a la etapa anterior; no se reduce agresividad de otras salvaguardas (checkpointing, `MIN_FREE_GIB`, exclusividad con Gradio) para compensar.
