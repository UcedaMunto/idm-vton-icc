# Version 10: perfilado por etapa y la mejor optimizacion encontrada hasta ahora

Fecha de creacion: 2026-08-22

## Estado

`OPTIMIZACION_ENCONTRADA_Y_VALIDADA_EN_20_PASOS`. Continuacion directa de V9: con
`--garmentnet_dtype=float32` ya corregido, Tier1 (`--hybrid_small_models_gpu`, de V7) seguia
siendo mas lento (201-204 s/paso) que el baseline 100% CPU (128 s/paso). V10 perfilo por etapa
para encontrar por que, y encontro que **`accelerate launch` fuerza `OMP_NUM_THREADS=1`**,
limitando todo el computo CPU (GarmentNet, UNet entrenable, VAE) a un solo nucleo de los 12
disponibles. Corrigiendo esto (`--cpu_threads=12`, nuevo flag) y combinandolo con Tier1, el
tiempo por paso baja a **93.93s** en frio (smoke test de 1 paso) y a **~55-58s en regimen
estacionario** (validado con una prueba de 20 pasos y 4 checkpoints, memoria estable de inicio a
fin). La mejor medicion de toda esta investigacion (V6-V10).

Ver [01_PERFILADO_Y_MEJOR_OPTIMIZACION.md](01_PERFILADO_Y_MEJOR_OPTIMIZACION.md) para el analisis
completo, tabla de mediciones y recomendacion final.

## Cambios aplicados a `train_xl.py` (ambos opt-in, sin cambiar el comportamiento por defecto)

1. `--profile_step_timing`: imprime tiempos por etapa de cada paso de entrenamiento
   (`[profile] <etapa>: <segundos>s`). Pura instrumentacion con `time.time()`.
2. `--cpu_threads=N`: si `N>0`, llama a `torch.set_num_threads(N)` tras crear el `Accelerator`,
   revirtiendo el `OMP_NUM_THREADS=1` que impone `accelerate launch`. Si no se pasa, no cambia
   nada respecto al comportamiento anterior a V10.

Ninguno de los dos flags cambia la logica de entrenamiento, la precision numerica objetivo, ni
donde se ubican los modelos (eso lo sigue controlando `--low_vram_training` /
`--hybrid_small_models_gpu`, ya validados en V7).

## Scripts de esta carpeta

- `config_v10.env`: config heredada de V7 (mismo checkpoint padre, misma resolucion), con
  `PROFILE_STEP_TIMING=1` agregado.
- `ejecutar_smoke_v10.sh`: smoke test base (Tier1 + `float32`, con perfilado).
- `ejecutar_smoke_v10_sin_checkpointing.sh`: variante sin `--gradient_checkpointing` (para medir
  su costo real; no recomendada para uso productivo por riesgo de swap, ver documento 01).
- `ejecutar_smoke_v10_baseline_cpu.sh`: variante sin Tier1 (100% CPU), para aislar el efecto del
  hilo unico sin la variable de Tier1 de por medio.
- `ejecutar_smoke_v10_thread_fix.sh`: baseline 100% CPU + `--cpu_threads=12`.
- `ejecutar_smoke_v10_combinado.sh`: Tier1 + `float32` + `--cpu_threads=12` (la mejor combinacion
  encontrada).
- `config_v10_larga.env` / `ejecutar_prueba_larga_v10.sh`: prueba de 20 pasos (checkpoints en
  5/10/15/20) con la combinacion recomendada, para validar estabilidad de RAM/swap mas alla de
  un solo paso.

Las pruebas de 1 paso son manuales y supervisadas, igual que en V4-V9. La prueba de 20 pasos
tambien es manual y supervisada (no se instala en cron), pero corre mas tiempo (~20 minutos).

## Que NO hace V10

- No cambia el comportamiento por defecto de `train_xl.py` para corridas que no pasen los nuevos
  flags explicitamente.
- No aplica todavia la combinacion recomendada a una corrida larga de produccion real (la prueba
  de 20 pasos reanuda desde un checkpoint de prueba de V4, no desde el checkpoint que el usuario
  elija para produccion).
- No repite el protocolo completo de comparacion de calidad de V6 (rubrica ciega); el fix de
  hilos y Tier1 no tocan precision numerica de forma material, pero la validacion final de
  calidad del modelo entrenado sigue pendiente segun el plan de V6.

## Documentos

1. [01_PERFILADO_Y_MEJOR_OPTIMIZACION.md](01_PERFILADO_Y_MEJOR_OPTIMIZACION.md): perfilado
   completo por etapa, hallazgo de `OMP_NUM_THREADS=1`, todas las mediciones (7 configuraciones
   comparadas), riesgo de memoria evaluado, y recomendacion final.
2. [02_PRUEBA_ESTABILIDAD_20_PASOS.md](02_PRUEBA_ESTABILIDAD_20_PASOS.md): prueba de 20 pasos
   con 4 checkpoints intermedios usando la combinacion recomendada; confirma estabilidad de
   RAM/swap y mide la velocidad real en regimen estacionario (~55-58s/paso, mejor que el smoke
   test en frio).
