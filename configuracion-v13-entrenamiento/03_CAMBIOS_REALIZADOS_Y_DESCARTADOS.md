# 03 - Cambios realizados y descartados (v11 -> v13)

Fecha: 2026-08-29.

## Cambios IMPLEMENTADOS y verificados

### V11 (causa raiz: repeticion de datos)
- `train_xl.py`: nueva clase `ResumableShuffleSampler` (barajado fijo y reproducible
  con semilla `--seed`, rotado para continuar donde quedo la cadena anterior) en
  lugar de `shuffle=False`.
- Seguimiento de `cumulative_steps` en el `manifest.json` de cada checkpoint
  compacto: al reanudar se lee la posicion del dataset y se persiste la nueva.
- Resultado: la cadena de bloques cortos ya avanza por el dataset en vez de
  repetir siempre las primeras ~500 parejas.

### V12 (observabilidad, continuidad y pruebas)
- `watchdog_entrenamiento.sh`:
  - Fix bug revision visual: `PYTHONPATH=<repo>` al invocar `comparar_calidad_v9.py`
    (antes fallaba con `ModuleNotFoundError: No module named 'src'`).
  - `RESUME_OPTIMIZER_STATE=1` (por defecto activo): pasa `--resume_optimizer_state`
    usando el `optimizer_state.pt` (3.2 GiB) que ya se persistia.
  - `LEARNING_RATE` (default 5e-5): permite experimentar sin editar el script.
- `configuracion-v12-entrenamiento/`:
  - `preservar_checkpoints_pruebas.sh` (+ cron cada 5 min): guarda los checkpoints
    intermedios antes de que el watchdog los borre al terminar el bloque.
  - `exportar_prueba.sh`: exporta un checkpoint preservado a un pipeline listo para
    la app. Se corrigio para usar el python de conda (`CONDA_PYTHON`) y `PYTHONPATH`
    (antes usaba `python3` del sistema sin diffusers y fallaba).
  - `.env` actualizado a `checkpoint 1800` para la prueba inicial con la app.
- Entrenamiento de produccion relanzado (bloque V12): resume optimizador + LR 5e-5,
  500 pasos por bloque. Se acumularon 1800 pasos y se exporto
  `result_train_v10/demos/prueba_cumulative_1800`.

## Intentado en V13 y DESCARTADO (importante)

### Limpieza de dataset por BRILLO -> DESCARTADO (incorrecto)
- Se creo `configuracion-v13-entrenamiento/limpiar_dataset.py` que filtraba pares
  cuya `cloth` no fuera "objeto sobre fondo blanco" usando un umbral de BRILLO
  (`gray < 228` para contar "no blanco").
- **Error detectado por el usuario**: ese criterio **descarta prendas blancas o muy
  claras** (una prenda blanca sobre fondo blanco tiene casi todo el encuadre
  brillante -> obj_frac ~0 -> clasificada como "VACIA"). Se confirmo: 000046 es
  VACIA real (std 8.4, sin bordes), pero 000065/000108/000133 son **prendas
  blancas VALIDAS** (std 12-13.6, con bordes).
- **Criterio descartado** (los archivos `*_pairs_clean.txt` generados se eliminaron
  y no se toco ninguna imagen del dataset).

### Deteccion de invalidas CORREGIDA (solo lectura, no aplicada)
Criterio recomendado para no perder prendas validas:
- 🧍 **PERSONA** (foto de persona como prenda): alto contenido de **piel**
  (`skin>0.15` y `obj>0.5`) -> descartar. Fiable.
- ⬜ **VACIA real** (imagen uniforme, sin silueta): **std muy bajo (<10) y bordes ~0**
  -> descartar. (Solo 000046 cumplio en la muestra.)
- 👕 **Prenda blanca/clara**: std bajo-medio PERO con bordes/silueta -> **CONSERVAR**.
- Recomendado: **curaduria visual** (lamina de contacto) en lugar de filtro
  automatico agresivo, dado que lo automatico es poco fiable con prendas claras.

## Recomendaciones pendientes de aplicar (no ejecutadas)

1. **Atacar el dominio** (la causa mas probable): preprocesar la prenda del usuario
   (aislar + fondo blanco, recortada, de frente) y/o dataset custom pequeno con sus
   prendas reales (adaptacion de dominio). Costo cero / alto impacto.
2. **Reducir la augmentacion de color**: bajar `hue` de 0.5 a ~0.1 y
   brightness/contrast/saturation a ~0.2 (experimento de una variable, protocolo V6).
3. **Moderar LR**: 2e-5 (mas seguro que 5e-5).
4. **Diagnostico A/B** (base vs 1800) sobre un par valido (p.ej. 048393) para
   confirmar si el color shift esta en la base (-> dominio) o lo introdujo el
   entrenamiento (-> receta).
5. **Alinear resolucion** entrenamiento/app (448x576 vs 576x768) y revisar
   `compute_time_ids` ((height,height) vs (height,width)).

## Estado

- Entrenamiento DETENIDO (watchdog pausado con PAUSAR_WATCHDOG).
- App corriendo en http://127.0.0.1:7860 con el checkpoint **1800**.
- Checkpoints preservados de 1200 a 1800 en
  `result_train_v10/pruebas_checkpoints/`.
- Dataset: intacto (no se modifico ninguna imagen). La limpieza automatica por
  brillo fue descartada; el criterio corregido (piel + uniformidad) esta
  documentado pero no aplicado.
