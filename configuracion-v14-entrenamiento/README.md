# Version 14: viabilidad comercial, dataset y plan de entrenamiento

Fecha: 2026-09-09 (actualizado 2026-09-10)
Estado: `EN_EJECUCION`. El paquete nacio como documentacion y plan; desde el
2026-09-10 incluye la primera implementacion (doc `10`). El entrenamiento tecnico
esta **detenido a proposito** el 2026-09-10 en **3.400 pasos acumulados**, con la
punta de la cadena preservada y verificada (ver doc `10.1`).

## AVISO LEGAL PRIMERO (leer antes de cualquier decision de negocio)

Con el **dataset actual** (VITON-HD) y la **base actual** (`yisol/IDM-VTON`) el
modelo resultante **NO es comercializable**. La cadena completa tiene licencias
no comerciales y el fine-tune es una obra derivada que las hereda:

| Pieza | Licencia | Uso comercial |
|---|---|---|
| Codigo IDM-VTON (`src/`, `train_xl.py`, `inference.py`, `gradio_demo/`) | CC BY-NC-SA 4.0 | NO |
| Checkpoints `yisol/IDM-VTON` (`result_train_v14/base_oficial`) | CC BY-NC-SA 4.0 | NO |
| Dataset VITON-HD (`dataset/DATA_DIR_PREP`) | CC BY-NC 4.0 "research only" | NO |
| DensePose (`ckpt/densepose/...`) | CC BY-NC 4.0 | NO |
| OpenPose (`ckpt/openpose/...`) | solo uso no comercial | NO |
| SCHP human parsing (`ckpt/humanparsing/...`) | MIT (codigo) | SI (con matices) |
| IP-Adapter (`ckpt/ip_adapter/...`) | Apache-2.0 | SI |
| SDXL base / inpainting | OpenRAIL++-M | SI (con restricciones) |

Detalle, fuentes citables y matices: `01_ANALISIS_LICENCIAS_Y_VIABILIDAD_COMERCIAL.md`.

**Consecuencia practica:** se puede seguir entrenando con el dataset actual
**solo como I+D no comercial** (Pista A). Para vender el modelo o usarlo dentro
de un producto o servicio hay que **sustituir las piezas no comerciales**
(Pista B, ver `02_COMPONENTES_Y_RUTA_A_MODELO_COMERCIAL.md`). Entrenar con datos
no comerciales y despues declarar los pesos como comerciales **no es una opcion
valida**: la restriccion viaja con los pesos derivados (obra derivada).

> Este paquete NO es asesoramiento legal. Es un analisis tecnico de licencias
> basado en los textos oficiales; la decision final debe revisarla un abogado.
> En `07_CHECKLIST_GO_NOGO_Y_RIESGOS.md` esta la lista de preguntas a resolver.

## Estado tecnico de partida (V14, 2026-09-09)

- Base de entrenamiento: `result_train_v14/base_oficial`, que es un **symlink**
  al snapshot oficial `yisol/IDM-VTON` en la cache de HuggingFace
  (`~/.cache/huggingface/hub/models--yisol--IDM-VTON/snapshots/585a32e7...`).
  Decision V13/04: partir SIEMPRE del oficial, nunca de `result_train_night`.
- Cadena de produccion: `result_train_v14/produccion_continua/` con un `run_*`
  por bloque de 500 pasos; logs en `logs/produccion_continua_v14/`.
- Receta activa del watchdog (`entrenamiento continuo/watchdog_entrenamiento.sh`):
  `--train_ip_adapter_only --hybrid_small_models_gpu --garmentnet_dtype=float32
  --cpu_threads=12 --gradient_checkpointing --low_vram_training
  --resume_optimizer_state`, LR **2e-5**, 448x576, batch 1, checkpoint cada 100.
- Progreso V14 acumulado (verificable en `logs/produccion_continua_v14/*.meta`):

  | Bloque | run_id | Inicio | Fin | Estado |
  |---|---|---|---|---|
  | 500 | `run_20260908_220420` | 2026-09-08 22:04 | 2026-09-09 04:07 | OK |
  | 1.000 | `run_20260909_041601` | 09-09 04:16 | 09-09 10:07 | OK |
  | 1.500 | `run_20260909_100801` | 09-09 10:08 | 09-09 15:55 | OK |
  | 1.800 | `run_20260909_160401` | 09-09 16:04 | - | interrumpido en el paso 300 |
  | 2.300 | `run_20260909_200600` | 09-09 20:06 | 09-10 04:06 | OK |
  | 2.800 | `run_20260910_041501` | 09-10 04:15 | 09-10 11:07 | OK |
  | 3.300 | `run_20260910_111601` | 09-10 11:16 | 09-10 18:00 | OK |
  | **3.400** | `run_20260910_180901` | 09-10 18:09 | 09-10 20:03 | **detenido a proposito** (checkpoint-100 guardado, paso 116) |

  Reanudacion: el `find` del watchdog coge `run_20260910_180901/checkpoint-100`,
  asi que **no se ha perdido nada**. El estado esta respaldado y verificado por
  sha256 en `backups/v14_estado_*/` (doc `10.1`) y se puede repetir con
  `respaldo_entrenamiento_v14.sh`.

- Checkpoints "compactos": `trainable_state.pt` (~1,7 GB) +
  `optimizer_state.pt` (~3,4 GB) + `manifest.json`. NO son pipelines cargables
  por la app; hay que exportarlos con `exportar_checkpoint_para_demo.py`.
- App web: `.env` -> `IDMVTON_MODEL_PATH=yisol/IDM-VTON` (modelo oficial, no los
  checkpoints entrenados). GPU de 12 GB: **no** pueden coexistir app y
  entrenamiento; el watchdog no lanza `train_xl.py` si detecta `gradio_demo/app.py`.
- Curva pasos -> fidelidad medida sobre las muestras reales del watchdog (ver
  `08_CURVA_PASOS_EFECTIVIDAD.md`): a **500 pasos acumulados** la prenda se
  transfiere bien; a **1.500** aparece el primer corrimiento de color (pantalones
  negros -> granate) y el modelo sigue derivando. A 1.800 la receta V12/V13 ya
  estaba rota (azul 0%, rojo 61-94%). Estimacion para **30.000 pasos: ~0-15%**
  (extrapolado, no medido). La palanca es la receta, no el numero de pasos.

## Resumen ejecutivo

1. **Tecnicamente** el proyecto funciona y es reproducible: cadena continua por
   watchdog, checkpoints validos, export a la app y A/B controlado.
2. **Comercialmente NO es viable tal cual** por la cadena de licencias NC
   (IDM-VTON + VITON-HD + DensePose/OpenPose). Es el hallazgo principal de V14.
3. Se definen **dos pistas**: Pista A (I+D no comercial con el dataset actual,
   legal hoy) y Pista B (ruta a un modelo comercial, exige sustituir piezas).
4. **Hardware actual**: RTX 3060 12 GB, 12 hilos (6 nucleos fisicos), 31 GiB de
   RAM, 87 GiB libres. Ritmo medido **~43 s/paso** con la maquina libre -> **~6 h
   por bloque de 500 pasos** (~2.000 pasos/dia en continuo). Un entrenamiento
   comercial serio (10.000-30.000 pasos) tarda **5-15 dias**; con el escritorio
   activo, **8-23 dias**. Ver `05_PLAN_TIEMPO_Y_HARDWARE.md`.
5. **Cuello de botella medido (ver `09_CUELLO_DE_BOTELLA_Y_HARDWARE.md`)**: el
   modelo **calcula en la CPU** por los flags activos (GPU al 10,9%, 49,3 W de
   170 W, 0 de 120 muestras por encima del 20%) y la **RAM no alcanza** (15 GiB
   de swap al 100%, iowait hasta 31%), asi que con el escritorio activo el paso
   sube de 42,8 a **65-87 s/paso**. El **disco** limita la **capacidad** (87 GiB
   -> ~10 bloques sin limpieza), no la velocidad. Compra recomendada en orden:
   RAM a 64 GB, despues GPU de 24 GB (o alquilar), y **no** CPU.

## Plan temporal resumido (detalle en `05_PLAN_TIEMPO_Y_HARDWARE.md`)

| Hito | Pasos | Solo computo | Con overhead | Bloques de 500 |
|---|---:|---:|---:|---:|
| Smoke | 100 | 1,2 h | 1,3 h | 1 |
| Piloto | 1.000 | 11,9 h | 12,2 h | 2 |
| MVP | 5.000 | 59,7 h (2,5 d) | 61,2 h (2,6 d) | 10 |
| Produccion corta | 10.000 | 119,4 h (5,0 d) | 122,4 h (5,1 d) | 20 |
| Produccion estandar | 20.000 | 238,9 h (10,0 d) | 244,9 h (10,2 d) | 40 |
| Produccion larga | 30.000 | 358,3 h (14,9 d) | 367,3 h (15,3 d) | 60 |
| 1 epoca (13.563 pares) | 13.563 | 162,0 h (6,8 d) | 166,2 h (6,9 d) | 28 |
| Receta original (130 epocas) | 1.763.190 | 21.060 h (877,5 d) | - | 3.527 |

> Valores generados con `python3 plan_tiempo_v14.py` (43,0 s/paso; overhead
> 0,15 h por bloque). La receta original es inviable en este hardware.

## Documentos

1. [01_ANALISIS_LICENCIAS_Y_VIABILIDAD_COMERCIAL.md](01_ANALISIS_LICENCIAS_Y_VIABILIDAD_COMERCIAL.md):
   cadena de licencias verificada con fuentes, veredicto y que implica "no comercial".
2. [02_COMPONENTES_Y_RUTA_A_MODELO_COMERCIAL.md](02_COMPONENTES_Y_RUTA_A_MODELO_COMERCIAL.md):
   opciones A/B/C, sustitucion pieza por pieza y arquitectura objetivo.
3. [03_DATASET_PARA_ENTRENAMIENTO_COMERCIAL.md](03_DATASET_PARA_ENTRENAMIENTO_COMERCIAL.md):
   dataset actual, formato que exige el codigo, requisitos y alternativas con licencia comercial.
4. [04_RECETA_ENTRENAMIENTO_COMERCIAL.md](04_RECETA_ENTRENAMIENTO_COMERCIAL.md):
   receta tecnica (que entrenar, augmentacion, LR, resolucion) y puertas de calidad A/B.
5. [05_PLAN_TIEMPO_Y_HARDWARE.md](05_PLAN_TIEMPO_Y_HARDWARE.md):
   throughput medido, escenarios temporales, calendario por fases, aceleraciones y coste.
6. [06_OPERACION_Y_RUNBOOK.md](06_OPERACION_Y_RUNBOOK.md):
   comandos de operacion (pausar/reanudar, exportar, cambiar modelo, monitorear, limpiar).
7. [07_CHECKLIST_GO_NOGO_Y_RIESGOS.md](07_CHECKLIST_GO_NOGO_Y_RIESGOS.md):
   checklist legal/tecnico, matriz de riesgos y decisiones abiertas.
8. [08_CURVA_PASOS_EFECTIVIDAD.md](08_CURVA_PASOS_EFECTIVIDAD.md):
   medicion de la curva pasos -> fidelidad de prenda y estimacion para 30.000 pasos.
9. `config_v14_comercial.env`: variables de configuracion documentadas.
10. `plan_tiempo_v14.py`: calculadora reproducible del plan temporal.
11. `analizar_curva_v14.py`: medicion reproducible de la curva (100% CPU).
12. [09_CUELLO_DE_BOTELLA_Y_HARDWARE.md](09_CUELLO_DE_BOTELLA_Y_HARDWARE.md):
    cuello de botella medido en vivo (CPU y RAM/swap), que hardware comprar y que no.
13. [10_PLAN_IMPLEMENTACION_V14.md](10_PLAN_IMPLEMENTACION_V14.md): estado real,
    fases F1-F5, calculo de VRAM de la receta C2 y gates A/B con comandos verificados.
14. `respaldo_entrenamiento_v14.sh`: respaldo del entrenamiento por hardlinks (0 bytes
    extra) con `MANIFEST.txt`, `verificacion.tsv` y modos `--verify` / `--list`.
15. `podar_disco_v14.sh`: retencion de disco con dry-run por defecto (protege el
    `resume_from` y nunca toca `base_oficial`, `demos/`, `experimentos/` ni `backups/`).
16. `experimento_v14.sh` + `config_v14_experimento.env`: banco de experimentos A/B
    aislado de produccion (`result_train_v14/experimentos/<tag>`, `--dry-run`).
17. `medir_ab_v14.py`: metrica de color del torso del A/B (la de V13) con rutas libres;
    validada contra las imagenes del A/B real de V13.

## Como usar este paquete

1. **Antes de cualquier decision de negocio**, leer `01` (bloqueo de licencias).
2. Si solo se quiere seguir investigando (legal hoy): seguir la **Pista A** con
   la receta de `04` y el calendario de `05`.
3. Si se quiere comercializar: abrir la **Pista B** (`02`) y el checklist de `07`.
4. Operacion diaria (pausar/reanudar/exportar/cambiar modelo): `06`.
5. Para ejecutar y medir el trabajo de V14: `10` (estado, fases F1-F5, gates A/B con
   comandos verificados) y los scripts `respaldo_entrenamiento_v14.sh`,
   `podar_disco_v14.sh`, `experimento_v14.sh` y `medir_ab_v14.py`.

## Relacion con las versiones anteriores

| Version | Aporte | Estado |
|---|---|---|
| V10 (`configuracion-v10-entrenamiento`) | Mejor optimizacion medida (Tier1 + float32 + 12 hilos) | Vigente (rendimiento) |
| V11 (`configuracion-v11-entrenamiento`) | Causa raiz: repeticion de datos; `ResumableShuffleSampler` | Vigente |
| V12 (`configuracion-v12-entrenamiento`) | Optimizer resume, revision visual, preservacion de checkpoints | Vigente |
| V13 (`configuracion-v13-entrenamiento`) | A/B controlado: el fine-tune degrada la transferencia; partir del oficial | Vigente (decision) |
| **V14 (este paquete)** | **Viabilidad comercial, dataset comercial y plan de tiempo; implementacion iniciada (doc 10)** | **En ejecucion** |

V14 **no** revierte ninguna decision tecnica anterior: mantiene la base oficial
(`result_train_v14/base_oficial`), la cadena `result_train_v14/produccion_continua`
y el watchdog con LR 2e-5, y anade la capa de negocio/licencias y el plan temporal.

## Historial

- 2026-09-09: creacion del paquete V14. Datos medidos de hardware, dataset y
  throughput; analisis de licencias verificado en fuentes oficiales; plan de
  tiempo y de viabilidad comercial.
- 2026-09-09: se anade `08_CURVA_PASOS_EFECTIVIDAD.md` y `analizar_curva_v14.py`:
  primera medicion de la curva pasos -> fidelidad de prenda sobre las muestras
  reales de V14 (500 y 1.500 pasos acumulados), calibrada contra el A/B de V13,
  con la estimacion para 30.000 pasos y sus limitaciones. Actualiza el bullet de
  estado del README.
- 2026-09-09: se anade `09_CUELLO_DE_BOTELLA_Y_HARDWARE.md`: diagnostico medido en
  vivo con el entrenamiento corriendo (`run_20260909_200600`). La GPU esta al 10,9%
  y 49,3 W de 170 W porque `--low_vram_training` + `--hybrid_small_models_gpu`
  dejan el UNet entrenable y GarmentNet en la CPU (`train_xl.py` 462, 479, 528-536,
  700-713, 1133); la RAM se queda corta (15 GiB de swap al 100%, iowait hasta 31%)
  y el paso pasa de 42,8 a 65-87 s/paso. Incluye 6 mejoras que no cuestan dinero y
  la prioridad de compra (RAM 64 GB > GPU 24 GB > NVMe; **no** CPU). Corrige el
  enfasis del bullet 5 del resumen ejecutivo (el disco limita capacidad, no
  velocidad) y actualiza el ritmo real observado.
- 2026-09-10: **inicio de la implementacion (doc `10`)**. App web detenida (puerto
  7860 libre, RAM de 21 a 4,6 GiB) y entrenamiento detenido a proposito en **3.400
  pasos**; punta de la cadena (`run_20260910_180901/checkpoint-100`) preservada y
  verificada por sha256, con respaldo por hardlinks en `backups/` (0 bytes extra) y
  herramienta reproducible (`respaldo_entrenamiento_v14.sh --verify`). Nuevo flag
  opt-in `--train_garmentnet` en `train_xl.py` y exportador del `unet_encoder` (antes
  un checkpoint C2 habria exportado el GarmentNet oficial sin avisar; el comparador
  A/B ahora aborta en vez de dar un gate incorrecto). **Hallazgo**: la receta C2
  completa necesita ~26-28 GB de VRAM (pesos + gradientes + 2 momentos de AdamW) y
  **no cabe en 12 GB**; quedan LoRA sobre el GarmentNet (al limite) o GPU de 24 GB /
  cloud. Se anaden el banco de experimentos A/B, el podador de disco (80 GiB
  liberables) y `medir_ab_v14.py` validado contra el A/B de V13.


