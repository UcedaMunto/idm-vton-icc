# Plan de comparacion de rendimiento y calidad

## Preguntas separadas

1. ¿La correccion operativa conserva el calculo del perfil actual?
2. ¿La reanudacion produce continuidad equivalente a una corrida sin corte?
3. ¿El perfil de 12 GB mantiene o mejora calidad frente a los controles?
4. ¿Un cambio matematico deliberado mejora calidad suficiente para promoverlo?

No se responde una pregunta con evidencia de otra.

## Controles

- `C0_original`: codigo de `/IDM-VTON-main` y parametros originales, cuando el hardware permita una prueba corta.
- `C1_pretrained`: `yisol/IDM-VTON`, sin entrenamiento local.
- `C2_checkpoint250`: `result_train_night/checkpoint-250`, actualmente usado en la app.
- `C3_current_hardware`: estado y parametros V4/V5 auditados.
- `V6_candidate`: una unica correccion sobre C3.

Cada resultado debe nombrar exactamente control, candidato, codigo, pesos y dataset.

## Conjunto de evaluacion

Crear una lista inmutable, fuera del train split usado en el experimento:

- Minimo de smoke: 6 pares para detectar fallos obvios.
- Comparacion de promocion: al menos 30 pares; recomendable 50-100.
- Incluir prendas lisas, estampadas, rayas, texto/logos, mangas cortas/largas y siluetas dificiles.
- Incluir diversidad de pose, oclusion, tonos de piel y fondos.
- Mantener persona, prenda, mascara, densepose, prompts y orden identicos.
- Usar semillas fijas multiples, por ejemplo 4 por par, para no decidir por una sola muestra favorable.

El conjunto, las entradas y sus hashes se congelan antes de ver resultados.

## Experimento A. Neutralidad operativa

Aplicable a locks, logs, parser, seleccion de rutas, checkpoint atomico y monitor.

1. Cargar exactamente los mismos pesos.
2. Ejecutar inferencia con mismos tensores, seed, scheduler, steps y precision.
3. Capturar salidas intermedias donde sea barato: embeddings, `time_ids`, primer `noise_pred`.
4. Exigir igualdad exacta si no se cambio backend/precision; si CUDA no es bit-determinista, definir tolerancia antes de ejecutar.

Criterio: ninguna diferencia visual y tensores dentro de tolerancia. Una diferencia obliga a reclasificar el cambio como numerico.

## Experimento B. Reanudacion exacta

Rama continua:

```text
estado S0 -> update 1 -> update 2 -> update 3 -> update 4 = S4_cont
```

Rama reanudada:

```text
estado S0 -> update 1 -> update 2 -> guardar/cerrar -> cargar -> update 3 -> update 4 = S4_resume
```

Comparar:

- indices y augmentaciones,
- ruido y timesteps,
- perdida por update,
- optimizer/scaler,
- parametros entrenables,
- snapshot de inferencia.

Primero exigir igualdad exacta en CPU determinista o tolerancia numerica predefinida en CUDA. Repetir eliminando optimizer/RNG para demostrar que la prueba detecta una reanudacion incompleta.

## Experimento C. Cambios matematicos

Para resolucion, BF16, clipping, optimizer, trainable scope o `time_ids`:

- Una variable por experimento.
- Mismo estado inicial y presupuesto de updates.
- Al menos tres replicas con semillas de entrenamiento distintas cuando el costo lo permita.
- Registrar tiempo/update, pico VRAM, pico RSS, disco, NaN/Inf y norma de gradiente.
- No usar solo loss para decidir calidad.

## Evaluacion automatica

Medidas recomendadas, siempre junto a inspeccion visual:

- LPIPS/DISTS entre persona original fuera de la zona editada y salida.
- SSIM/PSNR fuera de mascara para preservacion de identidad/fondo.
- CLIP o DINO similarity entre prenda de referencia y region vestida.
- Error de borde alrededor de la mascara.
- Deteccion de rostro/pose para fallos estructurales.
- KID/FID solo con un conjunto suficientemente grande; no reportarlo en smoke de 6 imagenes.

Guardar versiones de modelos metricos y codigo. Las metricas no deben descargarse o cambiar silenciosamente entre candidatos.

## Rubrica visual ciega

Presentar A/B con lado aleatorio y sin nombre de version. Puntuar 0-4:

- identidad/rostro,
- pose y anatomia,
- fidelidad de color,
- textura/patron/logo,
- forma, cuello y mangas,
- integracion en bordes,
- fondo y zonas no editadas,
- artefactos globales.

Registrar empates. Con un solo evaluador, repetir orden aleatorio en otra sesion; idealmente usar dos evaluadores.

## Criterios de no degradacion

Fijarlos antes del experimento. Propuesta inicial:

- Cero NaN/Inf, corrupcion o fallos de carga.
- Sin empeora grave nueva en ninguna categoria bloqueante: rostro, anatomia o prenda incorrecta.
- Candidato no pierde en mas del 10% neto de comparaciones visuales frente al control.
- Mediana de preservacion fuera de mascara no empeora mas del margen tecnico predefinido tras medir repetibilidad.
- Fidelidad de prenda no empeora significativamente; usar intervalo de confianza/bootstrap, no solo promedio.
- Uso de RAM/VRAM dentro de limites y sin swap destructivo.

Los umbrales numericos definitivos se calculan tras repetir C3 dos veces y medir variacion natural. No inventar tolerancias despues de ver al candidato.

## Registro por corrida

```text
run_id
line_id / parent_checkpoint_id
code_commit + dirty_diff_hash
config_requested + config_effective
base/compact/snapshot hashes
dataset/eval_set version
seed global + seeds por muestra
versions CUDA/PyTorch/Diffusers/Accelerate/xFormers/bitsandbytes
GPU y driver
updates acumulados y del bloque
loss, grad_norm, time/update, VRAM, RSS, disk
metricas y evaluacion visual
resultado: validated | candidate | rejected
motivo y responsable
```

## Regla de promocion

Un candidato solo reemplaza al control cuando pasa integridad, reproducibilidad, recursos y calidad. “Terminó sin OOM” no equivale a “no degrada el modelo”.
