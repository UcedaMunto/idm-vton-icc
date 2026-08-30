# Version 13: diagnostico documentado y correcciones propuestas

Fecha: 2026-08-29

## Estado

`DOCUMENTACION_DE_HALLAZGOS_Y_DECISIONES`. Este paquete recoge **todo lo
verificado con evidencia** en la investigacion (v11->v13), la **evaluacion de
las causas propuestas por ChatGPT**, y los **cambios realizados/rechazados**.
Ojo: NO se relanza el entrenamiento todavia; las correcciones de receta estan
pendientes de decision.

## Resumen ejecutivo

El sintoma reportado (prenda de salida con geometria generica, color desplazado
a rojo/naranja, textura/ruido sin identidad) **no lo causa un bug de
emparejamiento de datos** (se verifico que el loader toma cada archivo por el
nombre correcto), **ni un checkpoint mal cargado** (el export tiene todas las
claves y config identica a la que ya funcionaba), **ni entrenar desde SDXL base**
(la base checkpooint-250 es arquitectura IDM-VTON correcta: 13 canales +
ip_image_proj).

Lo que SI se confirmo con evidencia:

1. El modelo **apenas se mueve** con esta receta: deriva de pesos 0.68% (500),
   1.04% (1000), 1.10% (1100) pasos; y a 1800 pasos con LR 5e-5 -> 1.39% vs base.
   El modelo entrenado (1800) difiere ~0.3-0.6% del que usaba la app (1100).
2. La base local se comporta **igual que el oficial** `yisol/IDM-VTON` (V11),
   y como 1800 ~ base, el sintoma es en gran parte el **comportamiento inherente
   del modelo con esas entradas**, no un fallo del entrenamiento.
3. La receta es **debil para fidelidad de prenda**: entrenan solo IP-Adapter
   (14.17% de los parametros) y GarmentNet (la via de textura) esta congelado.
4. La **augmentacion de color** del dataset es agresiva
   (`ColorJitter(...hue=0.5)` = +-180 grados, identica al original) y es un
   candidato real al desplazamiento de color.
5. **LR 5e-5** NO es la causa principal (el sintoma existia antes de subir el LR,
   con el modelo 1100 entrenado a 1e-5), pero es un riesgo a vigilar.
6. Hay **datos "basura" en el dataset** (fotos de persona como prenda, imagenes
   uniformes), pero la deteccion automatica por BRILLO es **incorrecta** porque
   descarta prendas blancas/claras validas (ver 03).

## Implementacion de V13 (2026-08-29)

Cambios aplicados para el re-entrenamiento desde cero:

1. **Augmentacion de color reducida** (`train_xl.py`): `ColorJitter` ahora es
   configurable por argumentos y por defecto mas suave:
   `--color_jitter_prob=0.5 --color_jitter_brightness=0.2
   --color_jitter_contrast=0.2 --color_jitter_saturation=0.2
   --color_jitter_hue=0.1` (antes hue=0.5 = +-180 grados). Reduce la invariancia
   al color que colabora al desplazamiento de color.
2. **Learning rate moderado** (`watchdog_entrenamiento.sh`): `LEARNING_RATE` por
   defecto **2e-5** (antes 5e-5; 1e-5 no movia el modelo).
3. **Dataset limpio** (`generar_pares_limpios.py` + `VitonHDDataset`): el loader
   usa `{phase}_pairs_clean.txt` si existe. Solo descarta prendas
   inequivocamente malas (foto de persona por piel + imagen uniforme sin bordes);
   conserva prendas blancas/claras (tienen bordes/silueta). NO modifica imagenes.
4. **Cadena desde cero**: `watchdog_entrenamiento.sh` apunta a
   `result_train_v13/produccion_continua` (nuevo) y `logs/produccion_continua_v13`.
   Como la carpeta esta vacia, el find de resume no encuentra nada y arranca desde
   `result_train_night/checkpoint-250` (cumulative 0). La cadena V10 (1800) queda
   intacta y preservada.
5. `preservar_checkpoints_pruebas.sh` ahora sigue la cadena V13.

> Nota: el A/B diagnostico base-vs-1800 sigue pendiente; el entrenamiento V13 se
> lanza con la receta corregida para observar si mejora el comportamiento.

## Documentos

1. [01_HALLAZGOS_CON_EVIDENCIA.md](01_HALLAZGOS_CON_EVIDENCIA.md): todos los
   hallazgos verificados (numeros, rutas, codigo) de la investigacion.
2. [02_EVALUACION_CAUSAS_CHATGPT.md](02_EVALUACION_CAUSAS_CHATGPT.md): ranking de
   causas de ChatGPT evaluado con la evidencia de este proyecto (descarte/rebaja/
   confirmacion por causa).
3. [03_CAMBIOS_REALIZADOS_Y_DESCARTADOS.md](03_CAMBIOS_REALIZADOS_Y_DESCARTADOS.md):
   que se implemento (v11/v12), que se intento en v13 y se **descarto** (limpieza
   por brillo), y las recomendaciones corregidas.
