# 02 - Plan de receta de entrenamiento (una variable a la vez, protocolo V6)

Fecha: 2026-08-29

## Objetivo

Que el entrenamiento automatico mueva realmente el modelo (hoy mueve ~1% en
1100 pasos) y que se pueda VER el efecto en cada bloque (hoy la revision visual
esta rota). Regla V6: un cambio matematico por experimento, con validacion
visual fija antes de promover.

## Paso 0 (ya aplicado en V12): observabilidad y continuidad

- Fix del bug de revision visual del watchdog (PYTHONPATH). Sin impacto
  matematico.
- `--resume_optimizer_state` activado por defecto en el watchdog (usa el
  `optimizer_state.pt` de 3.2 GiB que ya se persistia en cada checkpoint).
  Impacto en RAM: ~3.4 GiB extra al reanudar (29 GiB disponibles; el pico de
  entrenamiento ~19-20 GiB RSS queda holgado). Unico cambio matematico de V12.
- Variable `LEARNING_RATE` en el watchdog (default 1e-5 = sin cambio) para poder
  experimentar sin editar el script.

## Paso 1: validar la continuidad del optimizador

1. Reanudar la cadena desde `run_20260825_190501/checkpoint-100`
   (cumulative_steps=1100) con `RESUME_OPTIMIZER_STATE=1`.
2. Criterio de exito: el `step_loss` del bloque nuevo arranca donde termino el
   anterior (~0.02-0.03) y no vuelve a saltar a ~0.04.
3. Verificar en el log la linea `[train] optimizer state resume enabled`.

## Paso 2: experimento de learning rate (no mezclar con otro cambio)

Opciones a probar, una a la vez, con un bloque de 500 pasos cada una y
comparando en el set fijo (ver Paso 4):

- A: LR 2e-5 (`LEARNING_RATE=2e-5`)
- B: LR 5e-5 (`LEARNING_RATE=5e-5`)

Fundamento: con solo IP-Adapter entrenable (423M params) y batch 1, las
recetas habituales de fine-tune de IP-Adapter usan LR entre 5e-5 y 1e-4.
1e-5 mueve los pesos ~0.7% por bloque de 500; 5e-5 deberia moverlos ~3.5x mas.
No subir de 1e-4 sin observar estabilidad (risk de oscilacion/olvido).

## Paso 3: presupuesto de pasos realista

- Meta sugerida: acumular 5000-10000 pasos (2.3-4.6 dias a ~41-58 s/paso) antes
  de evaluar calidad de forma seria. Con el sampler V11 eso cubre el 37-74% del
  dataset (1 pase parcial).
- NO evaluar calidad con 500-1200 pasos; a ese presupuesto la curva de loss ni
  siquiera se asento. Usar las imagenes de revision del watchdog (Paso 4) como
  control, no como veredicto.

## Paso 4: set fijo de evaluacion (debe existir antes de cualquier veredicto)

1. Elegir 10-30 parejas del split test con prendas reales aisladas (NO
   `048392_0`, cuya "prenda" es foto de persona). Guardar la lista con hashes.
2. Correr `comparar_calidad_v9.py` por cada candidato con la MISMA resolucion
   que la app (576x768) y semillas fijas (por ejemplo 4 semillas por pareja).
3. Comparar base vs candidato con la rubrica de V6
   (`configuracion-v6-entrenamiento/04_PLAN_COMPARACION_CALIDAD.md`): fidelidad
   de prenda, identidad/rostro, bordes, fondo.
4. Solo promover un checkpoint si pasa el criterio de no degradacion de V6.

## Paso 5 (despues, solo si hace falta)

- Alinear resolucion de la app con entrenamiento (o viceversa): hoy entrenamos a
  448x576 y la app genera a 576x768.
- Corregir `compute_time_ids()` para usar `(height, width)` en lugar de
  `(height, height)` (heredado del original; no es la causa principal pero es
  una inconsistencia entre entrenamiento e inferencia).
- Evaluar si conviene entrenar tambien parte de GarmentNet (unet_encoder), que
  es la via principal de textura de prenda y hoy esta 100% congelada.

## Decisiones que NO se tomaron en V12

- No se cambio la resolucion de entrenamiento (448x576) en este paso.
- No se cambio LR por defecto en el watchdog (sigue 1e-5; el experimento del
  Paso 2 se hace con la variable de entorno).
- No se toco `compute_time_ids` ni GarmentNet (requieren experimento separado).
- No se relanzo la cadena de produccion: el watchdog sigue pausado
  (`PAUSAR_WATCHDOG`); relanzarlo es decision del usuario.
