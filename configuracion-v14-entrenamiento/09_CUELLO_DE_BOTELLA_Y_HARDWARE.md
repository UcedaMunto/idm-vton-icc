# 09 - Cuello de botella real y mejora de hardware (medido en vivo)

Fecha: 2026-09-09, 22:15-22:25. Medicion hecha **con el entrenamiento corriendo**
(`run_20260909_200600`, paso ~120/500) y **sin tocarlo**: solo `nvidia-smi`,
`vmstat`, `iostat`, `/proc/*/io` y `smaps_rollup`. Reproducible con los comandos
de la seccion 9.8.

## 9.1 Respuesta corta

1. **El cuello de botella principal es la CPU, no la GPU.** El modelo calcula en
   la CPU por diseno de los flags activos (`--low_vram_training` +
   `--hybrid_small_models_gpu`): la GPU esta al **11% de uso y 49 W de 170 W**.
2. **El agravante es la RAM.** 31 GiB no alcanzan: hay **15 GiB de swap al 100%**
   (8 KiB libres), iowait de hasta **31%**, y el paso se ha degradado de
   **42,8 s/paso** (documentado en `05`) a **65,3 s/paso** medidos ahora, con el
   instantaneo subiendo de 73 a 92 s/it a lo largo de la corrida.
3. **El disco no limita la velocidad.** Su `%util` es 31-36% y sus lecturas son
   casi todas de *swap*, no del dataset. Limita la **capacidad** (87 GiB libres
   -> ~10 bloques antes de tocar `MIN_FREE_GIB`), no el ritmo.

## 9.2 Mediciones (2026-09-09 ~22:15, entrenamiento activo)

| Recurso | Medido | Lectura |
|---|---|---|
| GPU uso | **10,9% de media** en 120 muestras (min 5%, max 20%; **0 de 120** por encima del 20%) | ociosa ~89% del tiempo |
| GPU potencia | **49,3 W de 170 W** (un pico aislado de 104 W) | la GPU no es la que trabaja |
| GPU VRAM | 6,66 GiB de 12,0 GiB usados (~1,1 GiB son del escritorio) | libres **~5,3 GiB** |
| CPU | i5-10400, **6 nucleos fisicos / 12 hilos**, L3 de solo **12 MiB**, governor `powersave` | la que hace el trabajo |
| CPU proceso trainer | **976-978%** (de 1200%), 30 hilos, 21 h de CPU acumuladas | saturada |
| load average | **14,3** sobre 12 hilos | sobre-suscrita |
| iowait | **0,7% / 31,4% / 26,4%** (3 muestras de 2 s) | tramos de thrashing de swap |
| Disco | nvme0n1: 12,5 -> **496 -> 420 MB/s** de lectura, 58 KB por peticion, r_await 0,3 ms, **%util 31-36%** | no saturado; lee swap |
| RAM | 31 GiB totales, 23-24 GiB usados, **297 MiB libres**, 6-7 GiB de cache | al limite |
| Swap | **15 GiB / 15 GiB = 100%**, 8 KiB libres; vmstat con `so` de hasta 40.920 KB/s | thrashing activo |
| RSS del trainer (pid 613281) | **23,2 GiB** (Pss_anon 14,8 GiB) + **2,3 GiB en swap** | |
| RSS del worker del DataLoader (pid 613508) | 7,8 GiB + **13,0 GiB en swap** (duerme; fork con la misma cmdline) | duplica el costo de memoria |
| I/O acumulada del trainer | **690 GB leidos** en ~2 h | la cache no retiene nada |
| Ritmo real | **120 pasos en 2 h 10 m 40 s = 65,3 s/paso**; instantaneo 73,6 -> 81,7 -> **87,0 s/it** (paso 121) | vs. 42,8 s/paso documentado; y empeora |

Los dos procesos del entrenamiento suman **~31 GiB de RSS mas 15,3 GiB en swap**
sobre una maquina de 31 GiB compartida con el escritorio (Xorg, GNOME, Chrome,
VS Code, nautilus, eog: ~4 GiB y parte de los 12 hilos).

## 9.3 Por que: lo dice el propio `train_xl.py`

| Linea | Que hace | Consecuencia |
|---|---|---|
| 462 | `frozen_device = cpu if low_vram_training else accelerator.device` | los modelos congelados van a CPU |
| 479 | `image_proj_model` a CPU con `--low_vram_training` | idem |
| 700-713 | `unet.to("cpu")` + `accelerator.prepare(..., device_placement=[False]*5)` | el accelerator **nunca** mueve el modelo a la GPU |
| 528-536 | `--hybrid_small_models_gpu` mueve a GPU **solo** vae, text_encoder, text_encoder_2 e image_encoder | el UNet entrenable y GarmentNet se quedan en CPU |
| 530-531 | mensaje del log: *"GarmentNet and the trainable UNet remain on CPU"* | confirmacion en tiempo de ejecucion |
| 302, 458 | `--garmentnet_dtype` por defecto **float32** | GarmentNet (UNet SDXL, ~2,6B) en fp32 = **~10,4 GB de RAM** |
| 1133 | `unet_encoder(...)` dentro del bucle | el costo de GarmentNet se paga **en cada paso** |
| 417-418 | `torch.set_num_threads(args.cpu_threads)` -> `=12` | 12 hilos de GEMM sobre **6 nucleos fisicos** (HT) y 12 MiB de L3 |
| 557-560 | `cudnn.benchmark=False`, `allow_tf32=False` | las fases de GPU no se autotunean ni usan TF32 |
| 161-165 | la densepose se **lee precalculada** de `image-densepose/` | buena noticia: no hay DensePose en tiempo de paso |

En cada paso se ejecutan en la CPU **dos redes del tamano de un UNet de SDXL**
(el UNet entrenable en fp16 + GarmentNet en fp32, ~5,2B parametros en total) con
forward, backward y `--gradient_checkpointing` (que recalcula el forward para
ahorrar memoria, es decir gasta mas CPU a cambio de RAM). La GPU solo participa
en el VAE y en los codificadores CLIP: de ahi el 11% y los 49 W.

## 9.4 Por que hay swap: presupuesto de memoria

| Concepto | Tamano |
|---|---:|
| UNet entrenable (2,6B) en fp16 | ~5,2 GB |
| GarmentNet `unet_encoder` (2,6B) en **fp32** | **~10,4 GB** |
| IP-Adapter 423,8M: pesos fp16 + copia maestra fp32 + Adam m/v fp32 | ~5,1 GB |
| Activaciones (batch 1, 448x576, con gradient checkpointing) y varios | ~2 GB |
| Escritorio (Xorg, GNOME, navegador, VS Code, visores) | ~4 GB |
| **Total** | **~27-33 GB** |

El presupuesto supera los 31 GiB fisicos, asi que el swap es estructural, no un
sintoma de fuga. V10 ya lo habia documentado
(`configuracion-v10-entrenamiento/02_PRUEBA_ESTABILIDAD_20_PASOS.md`: "swap salta
a ~10-11 GiB tras el primer `optimizer.step()` y se queda ahi"). Hoy esta en
**15 GiB, el 100% de lo disponible**, porque la maquina comparte RAM con el
escritorio. Dicho de otro modo: **esta configuracion necesita mas RAM que la que
tiene la maquina**, y el sistema paga la diferencia con disco.

## 9.5 Impacto en el plan de tiempo de `05`

| Escenario | s/paso | 10.000 pasos | 30.000 pasos |
|---|---:|---:|---:|
| Documentado en `05` (maquina libre) | 42,8 | 5,1 dias | **14,9 dias** |
| Medido ahora (con escritorio y swap lleno) | 65,3 | 7,8 dias | **22,7 dias** |

Las cifras de `05_PLAN_TIEMPO_Y_HARDWARE.md` deben leerse como **el mejor caso**:
asumen la maquina sin carga adicional. Con el uso normal de este equipo el plan
realista es ~1,5x mas largo. Ademas la degradacion **crece con el tiempo de
corrida** (paso 120: 73,6 s/it; paso 121: 87,0 s/it) a medida que el swap se
llena, asi que 65,3 s/paso es un promedio, no un valor estable.

## 9.6 Mejoras sin comprar nada (hacerlas primero)

Ordenadas por relacion beneficio/riesgo. Todas son reversibles.

| # | Cambio | Efecto esperado | Riesgo |
|---|---|---|---|
| 1 | **Cerrar el escritorio** (navegador, VS Code, visores) o entrenar en modo texto | -4 GB de RAM, -20/-30% de los 12 hilos, -0,9 GB de VRAM | ninguno |
| 2 | `--garmentnet_dtype=bfloat16` | **-5,2 GB de RAM** (10,4 -> 5,2), menos ancho de banda de CPU | cambia la numerica: validar con el gate A/B |
| 3 | `--cpu_threads=6` (nucleos fisicos) | menos contencion de cache (L3 de 12 MiB) | neutro; medir A/B |
| 4 | Governor `powersave` -> `performance` | +5-10% de CPU | ninguno |
| 5 | Mover **solo GarmentNet** a la GPU en fp16 (~5,2 GB, cabe en los ~5,3 GiB libres) | saca 2,6B de parametros de la CPU | requiere un parche en la rama hybrid (linea 528-536); cambia numerica |
| 6 | `--checkpointing_steps=250` + podar `pruebas_checkpoints` | libera disco (capacidad, no velocidad); ver `06` | menos granularidad para A/B |

Las opciones 1, 3 y 4 no cambian el resultado del entrenamiento: son gratis.
Las opciones 2 y 5 si cambian la numerica y **deben** pasar por el gate A/B de
`04` antes de adoptarlas.

## 9.7 Que hardware comprar (y que no)

| Prioridad | Compra | Coste orientativo | Que desbloquea | Multiplicador esperado |
|---|---|---|---|---|
| **1** | **RAM a 64 GB** (2x32 GB DDR4; Comet Lake) | 70-130 USD/EUR | elimina el swap y el thrashing; recupera los 42,8 s/paso y probablemente mejor | 1,3-1,9x |
| **2** | **GPU de 24 GB** (RTX 3090 de segunda mano / 4090) | ~650-900 / ~1.800-2.200 | permite quitar `--low_vram_training`, `--gradient_checkpointing` y `--hybrid_small_models_gpu`: todo el modelo en GPU | 4-6x (cifra orientativa de `05`, a medir) |
| 3 | NVMe de 2 TB dedicado a checkpoints | 110-160 | quita el limite de ~10 bloques y la presion del cron; saca swap y checkpoints del disco del sistema | capacidad, no velocidad |
| - | **CPU** | - | **no comprar**: mas nucleos no arreglan que el trabajo este donde no debe | ~0 |
| - | Fuente de alimentacion: si se compra GPU de 24 GB hace falta 750-850 W | 90-130 | requisito de la GPU | - |

Razonamiento:

- **La RAM primero.** Es la compra mas barata, la que mas se nota hoy, y tambien
  la que hace falta para cualquier otro escenario. Con 64 GB desaparecen los
  15 GiB de swap, el iowait y los 690 GB de lecturas repetidas.
- **La GPU despues, y solo para entrenar en serio.** Con 24 GB los pesos caben
  enteros en VRAM y desaparece todo el offload. Antes de comprarla, medir con
  `05_PLAN_TIEMPO_Y_HARDWARE.md`: si el objetivo es un entrenamiento puntual,
  **alquilar sale mas barato** (30.000 pasos en A100: ~100-250 USD, frente a
  15-23 dias de esta maquina). Comprar solo si el uso es continuo.
- **El disco solo por capacidad.** Ya es un NVMe con 0,3 ms de latencia y 36% de
  utilizacion: no es el freno.
- **El CPU es la pieza que hoy trabaja, y aun asi no hay que comprarla.** Gastar
  en un CPU de 16 nucleos para ejecutar un UNet es invertir en la capa
  equivocada: el mismo dinero en una GPU hace ese trabajo entre 10 y 50 veces
  mas rapido.

## 9.8 Reproducir esta medicion

```bash
cd /home/uceda/Documents/IDM-VTON

nvidia-smi                                  # uso, potencia y VRAM de la GPU
for i in $(seq 1 120); do                   # reparto de uso de la GPU en un paso
  nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader
  sleep 0.4
done

vmstat 2 3                                  # runqueue (r) y swap (si/so)
iostat -x 2 3                               # iowait y utilizacion real del disco
free -h ; cat /proc/loadavg
for p in $(pgrep -f train_xl.py); do        # quien gasta la memoria
  grep -E 'VmRSS|VmSwap' /proc/$p/status
done
```

## 9.9 Limitaciones de esta medicion

1. **Es una sola foto** de 10 minutos tomada en un momento concreto, aunque las
   120 muestras de GPU cubren un paso completo y son muy estables (5-20%,
   media 10,9%).
2. **La maquina tenia el escritorio activo** durante la medicion (navegador,
   VS Code, visores). Es el estado realista de uso, pero no el de referencia de
   `05`, que se midio con la maquina libre.
3. **La degradacion de 42,8 a 65,3 s/paso esta correlacionada, no demostrada.**
   Los candidatos medidos son la contencion con el escritorio y el thrashing de
   swap (15 GiB al 100%, iowait hasta 31%). Para convertirla en causa hay que
   medir un bloque con el escritorio cerrado: es gratis y es el proximo paso.
4. **Los precios son orientativos** (2026) y no se han verificado con
   proveedores locales; hay que confirmarlos antes de presupuestar.
5. Los ahorros de 9.6 (opciones 2 y 5) cambian la numerica del entrenamiento y
   no estan validados con el gate A/B de calidad.


