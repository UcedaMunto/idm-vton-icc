# 10 - Plan de implementacion V14 (de la documentacion a la ejecucion)

Fecha: 2026-09-10. Estado: `EN_EJECUCION`.

Los documentos 01-09 de este paquete **describen y planifican**. Este documento
convierte ese plan en trabajo ejecutable: que se ha hecho ya, que se puede hacer sin
comprar nada, que exige hardware y que exige una decision de negocio.

## 10.1 Estado de partida (hecho y verificado el 2026-09-10)

| Accion | Evidencia |
|---|---|
| App web detenida | procesos `gradio_demo/app.py` (pids 727041/727081) terminados; puerto 7860 libre (`ss -ltnp`); RAM usada 21 GiB -> **4,6 GiB**; VRAM 998 MiB -> 794 MiB (escritorio) |
| Entrenamiento detenido a proposito | no hay `train_xl.py`; `PAUSAR_WATCHDOG` presente desde 20:03; **no** hay `PAUSAR_POR_ERROR` |
| Cadena preservada y verificada | punta `produccion_continua/run_20260910_180901/checkpoint-100` + ultimo bloque completo `run_20260910_111601/checkpoint-500`; `trainable_state.pt` origen y copia de `pruebas_checkpoints` con **sha256 identico** (`8c7ea43a...`) |
| Respaldo con coste 0 | `backups/v14_estado_20260910/` por **hardlinks** (link count 2, mismos inodos): disco sin cambios (518G/48G libres antes y despues) |
| Respaldo reproducible | `respaldo_entrenamiento_v14.sh` (+ `--verify` por sha256) |

Nota sobre el "coste 0" del respaldo: `du -sh` informa ~9,5 GiB, pero son los mismos
inodos que los checkpoints originales (los hardlinks no duplican datos). El espacio real
anadido es el de los ficheros de texto (MANIFEST + verificacion.tsv) y la entrada de
directorio; la prueba es que `df` no cambia antes y despues. La contrapartida es que el
respaldo solo sobrevive mientras existan los datos compartidos: si se borra el original,
el inodo sigue vivo por el hardlink, pero si se **sobrescribe** un fichero in-place,
ambos cambian. En este proyecto los checkpoints se crean y se borran por directorio
completo, nunca se reescriben, asi que el riesgo es nulo.

Progreso acumulado de la V14 (verificable en `logs/produccion_continua_v14/*.meta`):

| Pasos | run_id | Inicio | Fin | Estado |
|---|---|---|---|---|
| 500 | `run_20260908_220420` | 09-08 22:04 | 09-09 04:07 | OK |
| 1.000 | `run_20260909_041601` | 09-09 04:16 | 09-09 10:07 | OK |
| 1.500 | `run_20260909_100801` | 09-09 10:08 | 09-09 15:55 | OK |
| 1.800 | `run_20260909_160401` | 09-09 16:04 | - | interrumpido en el paso 300 |
| 2.300 | `run_20260909_200600` | 09-09 20:06 | 09-10 04:06 | OK |
| 2.800 | `run_20260910_041501` | 09-10 04:15 | 09-10 11:07 | OK |
| 3.300 | `run_20260910_111601` | 09-10 11:16 | 09-10 18:00 | OK |
| **3.400** | `run_20260910_180901` | 09-10 18:09 | detenido a proposito | `checkpoint-100` guardado (paso 116) |

Reanudar = quitar `PAUSAR_WATCHDOG` y lanzar el watchdog: el `find` de resume coge
`checkpoint-100` y la cadena sigue en 3.400. **El progreso no se ha perdido.**

## 10.2 Hallazgo que cambia el plan: la receta C2 no cabe en 12 GB

El doc 04 recomienda la opcion **C2** (entrenar IP-Adapter **+ GarmentNet**) porque es
el GarmentNet quien aporta textura/geometria de la prenda, y el doc 09 (§9.6, opcion 5)
apunta que "mover solo GarmentNet a la GPU en fp16 (~5,2 GiB) cabe en los ~5,3 GiB
libres". Eso vale para **inferencia** (solo forward). Para **entrenar** hay que sumar
gradientes y estado del optimizador:

| Pieza | Tamano | Nota |
|---|---|---|
| Pesos GarmentNet en fp16 | ~5,2 GB | 2,6B parametros x 2 B |
| Gradientes en fp16 | ~5,2 GB | se liberan al hacer `zero_grad` |
| Estado de AdamW (2 momentos, fp16) | ~10,4 GB | `torch.zeros_like(p)`: mismo dtype que el parametro |
| Activaciones (gradient checkpointing, batch 1, 448x576) | ~2-4 GB | estimado |
| pila ya existente de la app/trainer (vae + 2 text encoders + image encoder en fp16) | ~5,5 GB | medido en el doc 09 |

**Total minimo ~26-28 GB** frente a **12 GB** de VRAM: la receta C2 completa **no es
viable en esta maquina**, ni de lejos. No es un problema de ajuste fino.

Alternativas, ordenadas por coste:

| Camino | Que exige | Viabilidad en 12 GB |
|---|---|---|
| C2-lite: **LoRA/adapters** sobre el GarmentNet (no pesos completos) | implementar LoRA en `src/unet_hacked_garmnet` (peft) y dejar el GarmentNet congelado en GPU | **al limite**: ~5,2 (GN congelado) + ~0,5 (LoRA + optimizador) + activaciones + pila; solo cerrando el escritorio y renunciando a parte del tier hibrido. Hay que **medirlo**, no estimarlo |
| C2 con **Adam de 8 bits** (`--use_8bit_adam`) | quitar el veto a 8-bit Adam cuando se entrena el GarmentNet (hoy se desactiva con `--low_vram_training`) y tener `bitsandbytes` funcionando | sigue sin caber: 5,2+5,2+2,6+activaciones+pila ≈ 19-21 GB |
| **GPU de 24 GB** (doc 09, prioridad 2) | compra/alquiler | C2 completa cabe si el GarmentNet se entrena con la pila pequena en CPU (~17 GB) |
| **Alquiler en cloud** (doc 09) | presupuesto | para un entrenamiento puntual sale mas barato que comprar |

> **Esto es aritmetica, no una medicion.** El orden de magnitud es solido (los pesos y
> los dos momentos son calculos exactos), pero las activaciones y la pila son estimados.
> Se mide con el smoke test: `bash experimento_v14.sh --tag c2_smoke --steps 1` con
> `TRAIN_GARMENTNET=1` y siguiendo la VRAM con `nvidia-smi --query-gpu=memory.used --format=csv -l 5`.
>
> Consecuencia practica: **el entrenamiento productivo no debe seguir acumulando pasos
> de la receta C1** (V13 demostro que degrada la transferencia y el gate del doc 04 no
> esta pasado). Antes de gastar mas horas hay que decidir entre la via LoRA (barata pero
> hay que implementarla) o el hardware/cloud de la via C2.

## 10.3 Que se ha implementado hoy (F1) - sin cambiar la numerica

Todo lo de esta seccion es **reversible** y **no altera el entrenamiento actual**: con
los valores por defecto el comportamiento es identico al de antes.

| Artefacto | Que hace |
|---|---|
| `train_xl.py --train_garmentnet` (nuevo, opt-in) | hace entrenable el GarmentNet: `requires_grad`, lo sube a GPU (no puede quedarse en CPU), lo mete en el optimizador, lo pone en modo `train()` y lo guarda/reanuda. **Sin el flag no cambia nada** |
| `exportar_checkpoint_para_demo.py` | ahora entiende el prefijo `unet_encoder.` y regenera la subcarpeta `unet_encoder/` del pipeline. **Antes la symlinkeaba al modelo base**: un checkpoint C2 habria exportado el GarmentNet oficial sin avisar |
| `respaldo_entrenamiento_v14.sh` | respaldo por hardlinks (0 bytes) + `MANIFEST.txt` + `verificacion.tsv` con sha256 + modos `--verify` y `--list` |
| `podar_disco_v14.sh` | retencion de disco con dry-run por defecto; protege el run del `resume_from` y nunca toca `base_oficial`, `demos/`, `experimentos/` ni `backups/` |
| `experimento_v14.sh` + `config_v14_experimento.env` | lanzador de experimentos A/B aislado (`result_train_v14/experimentos/<tag>`) con `--dry-run`, comprobacion de pausa/VRAM/disco y `.meta` reproducible |
| `10_PLAN_IMPLEMENTACION_V14.md` | este documento |

Verificado en vivo: `py_compile` de los dos python, `train_xl.py --help` muestra el
flag nuevo, el exportador importa el GarmentNet, `bash -n` de los tres scripts y
dry-run del experimento y del podador ejecutados sin tocar nada.

## 10.4 Fases

| Fase | Contenido | Estado | Criterio de aceptacion |
|---|---|---|---|
| **F1** Operacion y seguridad | respaldo, retencion, runbook | **HECHA** | `respaldo_entrenamiento_v14.sh --verify` sin fallos y `podar_disco_v14.sh` en dry-run |
| **F2** Gate A/B de lo ya entrenado (3.400 pasos) | exportar la punta y comparar contra el oficial | **LISTA, no lanzada** | doc 04: torso azul >50% y sin artefactos rojo/magenta |
| **F3** Limpieza de la receta por A/B | un experimento por variable: `hue=0.0`, `garmentnet_dtype=bfloat16`, `cpu_threads=6`, resolucion 576x768 | pendiente | cada variable que no mejore el gate se descarta |
| **F4** Entrenar el GarmentNet (C2) | o LoRA sobre el GarmentNet (12 GB, al limite) o GPU de 24 GB / cloud | **bloqueada por hardware** | tiene que caber: medirlo con el smoke y `nvidia-smi` |
| **F5** Pista B (comercial) | stack limpio + dataset propio o licenciado (docs 02 y 03) | requiere decision y presupuesto | checklist del doc 07 |

## 10.5 Gates A/B: comandos verificados

Protocolo del doc 04 (4.6): par de control `048400_0` (prenda de catalogo azul), seed 42,
mismo width/height y 15 pasos para las dos corridas. Criterio de aceptacion:
**torso azul > 50%** y sin artefactos rojo/magenta.

```bash
cd /home/uceda/Documents/IDM-VTON
export PYTHONPATH=/home/uceda/Documents/IDM-VTON   # sin esto el comparador no importa src
PY=/home/uceda/miniconda3/envs/idm/bin/python

# 0) El par de control 048400_0 esta en la LINEA 9 de test_pairs.txt (no es el
#    primero: los primeros son 048392_0, 048393_0...). Se prepara un data_dir
#    temporal con esa unica linea: el dataset original no se toca.
mkdir -p /tmp/ab14/data
ln -sfn "$PWD/dataset/DATA_DIR_PREP/test" /tmp/ab14/data/test
sed -n '9p' dataset/DATA_DIR_PREP/test_pairs.txt > /tmp/ab14/data/test_pairs.txt
cat /tmp/ab14/data/test_pairs.txt                  # 048400_0.jpg 048400_0.jpg

# 1) REFERENCIA = modelo OFICIAL (sin fine-tune).
$PY configuracion-v9-entrenamiento/comparar_calidad_v9.py \
  --pretrained_model_name_or_path result_train_v14/base_oficial \
  --data_dir /tmp/ab14/data --output_dir /tmp/ab14/oficial \
  --width 448 --height 576 --num_inference_steps 15 --seed 42 --limit 1

# 2) CANDIDATO = punta de la cadena V14. No hace falta exportar: el comparador
#    superpone el checkpoint compacto con --compact_checkpoint.
$PY configuracion-v9-entrenamiento/comparar_calidad_v9.py \
  --pretrained_model_name_or_path result_train_v14/base_oficial \
  --compact_checkpoint result_train_v14/produccion_continua/run_20260910_180901/checkpoint-100 \
  --data_dir /tmp/ab14/data --output_dir /tmp/ab14/v14_cum3400 \
  --width 448 --height 576 --num_inference_steps 15 --seed 42 --limit 1

# 3) Metrica de color del torso (la de V13, pero con rutas libres y veredicto).
$PY configuracion-v14-entrenamiento/medir_ab_v14.py \
  --referencia /tmp/ab14/oficial/048400_0.jpg \
  --candidato  /tmp/ab14/v14_cum3400/048400_0.jpg \
  --prenda dataset/DATA_DIR_PREP/test/cloth/048400_0.jpg \
  --etiqueta-referencia OFICIAL --etiqueta-candidato v14_cum3400
```

El nombre del fichero de salida es el de la imagen de entrada (`048400_0.jpg`). Cada
corrida de inferencia tarda varios minutos y consume ~10 GiB de RAM: **no lanzar las dos
a la vez** en esta maquina.

### Nota critica para la receta C2

`comparar_calidad_v9.py` **solo** superpone el unet. Con un checkpoint de la receta C2
(los que traen pesos del GarmentNet) **aborta con un mensaje explicito** en vez de dar un
gate incorrecto. Para esos:

```bash
# Exportar (fusiona unet + unet_encoder) y comparar contra la carpeta exportada.
$PY exportar_checkpoint_para_demo.py --base_checkpoint result_train_v14/base_oficial \
  --compact_checkpoint <ckpt_c2> --output_dir /tmp/ab14/export_c2
$PY configuracion-v9-entrenamiento/comparar_calidad_v9.py \
  --pretrained_model_name_or_path /tmp/ab14/export_c2 \
  --data_dir /tmp/ab14/data --output_dir /tmp/ab14/c2 \
  --width 448 --height 576 --num_inference_steps 15 --seed 42 --limit 1
```

### La metrica esta validada contra el A/B real de la V13

`medir_ab_v14.py` se ejecuto sobre las imagenes reales de
`configuracion-v13-entrenamiento/resultado_ab_20260830/salidas/` y reproduce sus numeros
documentados:

| imagen | azul% | rojo% | veredicto |
|---|---|---|---|
| prenda de entrada | 100,0% | 0,0% | (referencia: la prenda es azul) |
| `base_heredada` | 0,3% | 61,3% | FALLA |
| `v13_cum800` | 0,1% | 72,6% | FALLA |

Es exactamente la conclusion del doc 04 (el fine-tune actual no transfiere la prenda),
asi que la metrica que se usara en los gates de V14 es la misma que decidio V13.

```bash
cd /home/uceda/Documents/IDM-VTON
CKPT=result_train_v14/produccion_continua/run_20260910_180901/checkpoint-100
OUT=result_train_v14/demos/v14_cum3400

# 1) Exportar la punta de la cadena a un pipeline cargable (~10 min; OUT no debe existir).
PYTHONPATH=/home/uceda/Documents/IDM-VTON /home/uceda/miniconda3/envs/idm/bin/python \
  exportar_checkpoint_para_demo.py \
  --base_checkpoint result_train_v14/base_oficial \
  --compact_checkpoint "$CKPT" \
  --output_dir "$OUT"

# 2) Servirlo en la app (opcional) y reiniciar para aplicar el cambio.
./switch_model_version.sh custom "$PWD/$OUT"

# 3) Comparar contra el oficial con el protocolo del doc 04 (par de control 048400_0,
#    seed 42, 448x576, 15 pasos) y medir el color del torso con la metrica de V13.
/home/uceda/miniconda3/envs/idm/bin/python configuracion-v9-entrenamiento/comparar_calidad_v9.py --help
/home/uceda/miniconda3/envs/idm/bin/python configuracion-v13-entrenamiento/resultado_ab_20260830/ab_analisis.py --help
```

Los comandos exactos de cada uno estan en `configuracion-v13-entrenamiento/04_RESULTADO_AB.md`
(protocolo ya validado en V13) y en el doc 06 de esta carpeta.
