# Auditoria de cambios y posibles fallos

Fecha: 2026-08-22

## Metodo y limites

Se compararon el contenido de `/home/uceda/Documents/IDM-VTON-main`, el historial Git del proyecto modificado, el `train_xl.py` no confirmado y las carpetas V2-V5. No se ejecuto un entrenamiento A/B: las conclusiones sobre calidad son riesgos tecnicos que deben validarse con el protocolo V6, no resultados experimentales.

Escala:

- `BLOQUEANTE`: puede reanudar una linea equivocada, perder continuidad o automatizar fallos.
- `ALTO`: cambia directamente la trayectoria matematica o invalida la comparacion.
- `MEDIO`: reduce reproducibilidad, trazabilidad o robustez.
- `BAJO`: inconsistencia operativa sin efecto matematico directo demostrado.

## Resumen por etapa

| Etapa | Cambio principal | Resultado |
|---|---|---|
| Original | Entrenamiento SDXL de referencia | Tiene defectos heredados: conteo doble, checkpoint al final de epoca, `time_ids` con ancho incorrecto y rama `sample` invalida. |
| Git `986bafc` | Configuracion/documentacion de entrenamiento | Principalmente configuracion; no resuelve todavia control de steps. |
| Git `240b9ba` | Perfil 12 GB y reduccion de RAM | Introduce low-VRAM, entrenamiento parcial, casts y cambios de optimizador/precision. No es matematicamente equivalente. |
| Git `105ece7` | Launcher largo | Parametriza corridas, pero hereda el perfil alterado. |
| Git `89aa18d` | Seleccion de modelo entrenado | Permite cargar checkpoint local en Gradio/inferencia; no cambia pesos por si mismo. |
| V2 | Diagnostico | Identifica correctamente conteo doble, checkpoint tardio y mezcla de lineas. |
| V3 | Checkpoint compacto | Corrige steps y frecuencia, pero no guarda un estado completo de continuidad. |
| V4 | BF16 y menor RAM | Reduce RAM, omite optimizer al reanudar y cambia numerica. |
| V5 | Automatizacion documentada | La implementacion auditada no coincide con lo documentado. |

## Erratum de auditoria (2026-08-22, revision posterior)

El hallazgo F-01 original (mas abajo, tachado en su contenido literal) se redacto a partir de una lectura incompleta de `entrenamiento continuo/watchdog_entrenamiento.sh` obtenida con la herramienta de lectura de archivos, que devolvio una version truncada/desactualizada del contenido (probablemente por la ruta con espacios). Al forzar una prueba real -deteniendo un entrenamiento activo con SIGTERM- se confirmo que:

- El script SI carga una unica fuente de configuracion: `CONFIG_FILE` (por defecto `configuracion-v4-entrenamiento/config_v4.env`) mediante `source`, con `set -a`/`set +a`.
- El script SI implementa un supervisor real: lanza `accelerate launch train_xl.py` de forma sincrona dentro de una subshell en segundo plano, captura `EXIT_CODE=$?`, valida la existencia de `checkpoint-<max_train_steps>/manifest.json` y `trainable_state.pt`, y si falta o el exit code no es cero escribe `PAUSAR_POR_ERROR` de forma atomica con `run_id`, `exit_code`, `expected_checkpoint`, `log_file` y `failed_at`.
- Se verifico en vivo: al enviar `SIGTERM` a la corrida `v4_run_20260822_104901`, el supervisor escribio `PAUSAR_POR_ERROR` con `exit_code=143` en menos de un minuto, sin intervencion de cron.
- El hash del archivo (`sha256sum`) fue identico entre la auditoria original y esta revision (`6fdc4e4c5bee62df959d095f082a10cf6cf9b02f7f87e2657e576fda42b56850`), confirmando que el archivo no cambio: el error fue de lectura, no de contenido.

Conclusion corregida: la automatizacion V4/V5 es mas robusta de lo que la auditoria original describio. F-01 se reclasifica de `BLOQUEANTE` a `MEDIO` y se redefine mas abajo con el alcance real que aun falta cerrar.

## Hallazgos bloqueantes

### F-01 (reclasificado a MEDIO tras verificacion). Brechas reales del supervisor existente

Lo que SI funciona (confirmado en vivo):

- Config unica via `config_v4.env`.
- Supervisor sincrono que espera el proceso y valida `manifest.json` + `trainable_state.pt`.
- Disyuntor atomico ante exit code distinto de cero o checkpoint incompleto.

Lo que sigue faltando (brechas reales, no el mismatch original):

- La validacion del checkpoint solo comprueba existencia de dos archivos, no su integridad, hash, ni compatibilidad de linaje con `CONTINUATION_CHECKPOINT`/`RESUME_CHECKPOINT`.
- `config_v4.env` se ejecuta con `source` (shell real), no con un parser declarativo; un valor malformado puede ejecutar codigo arbitrario.
- El lock (`flock`) solo cubre la vida del proceso lanzador del watchdog, no la corrida en si; la proteccion contra relanzamientos duplicados depende de `pgrep -fa train_xl.py`, que es un guardia razonable pero no atomico.
- El manifest no referencia un `parent_checkpoint_id` ni hash de codigo, por lo que sigue sin poder demostrarse linaje formal (ver F-02).

Correccion V6 planificada: mantener el supervisor actual como base; anadir validacion de linaje/hash del checkpoint, migrar `config_v4.env` a un formato declarativo no ejecutable para la interfaz, y anadir prueba de fallo intencional repetible como parte de la promocion a V6.

### F-02. Seleccion de checkpoint por fecha, no por cadena validada

El watchdog busca el `trainable_state.pt` mas reciente por mtime. Solo comprueba que haya un `manifest.json`; no valida base, hash del codigo, resolucion, conjunto entrenable, numero acumulado, integridad ni estado de aprobacion.

Impacto: puede combinar un pipeline base y pesos compactos de experimentos diferentes. Coincidir en nombres de tensores no demuestra compatibilidad semantica.

Correccion V6 planificada: identificador inmutable de linea, `parent_checkpoint_id`, hashes, esquema de argumentos y promocion explicita `validated=true`.

### F-03. Reanudacion no equivale a continuidad

El checkpoint compacto guarda pesos entrenables y optimizer, pero no guarda RNG de Python/PyTorch/CUDA, sampler/dataloader, posicion de epoca, scaler AMP ni contador acumulado. V4/V5, ademas, no carga optimizer por defecto. Si se pide `--resume_optimizer_state` y falta el archivo, el codigo sigue y anuncia que la opcion esta habilitada sin confirmar una carga.

Impacto: dos bloques de un update no equivalen a dos updates continuos. AdamW reiniciado cambia momentos y actualizaciones; las augmentaciones y muestras VAE cambian al no restaurar RNG.

Correccion V6 planificada: dos modos con nombres no ambiguos: `continue_exact` exige estado completo o falla; `warm_start_weights` inicia una nueva fase y nunca se presenta como reanudacion exacta.

### F-04. No existe snapshot de inferencia promovible en la cadena compacta

Gradio carga un pipeline con `from_pretrained`; el checkpoint compacto solo contiene `.pt` y manifest. `full_pipeline_checkpoint` es opcional y el perfil V4 no lo activa.

Impacto: entrenar puede producir estados recuperables que no se pueden evaluar directamente en la aplicacion. Esto favorece seguir evaluando `checkpoint-250` mientras otra linea avanza.

Correccion V6 planificada: separar estado de entrenamiento y snapshot de inferencia, enlazarlos por ID/hash y promover solo snapshots construidos y validados.

## Hallazgos altos que afectan el modelo

### F-05. Se cambio el conjunto de parametros entrenables

El original entrena el UNet completo. El perfil usa `--train_ip_adapter_only`: congela el UNet y deja entrenables attention processors/IP-Adapter, `encoder_hid_proj` y `conv_in`.

Impacto: cambia el objetivo de optimizacion y la capacidad adaptativa. Es una estrategia legitima para 12 GB, pero no una correccion neutra.

### F-06. Se cambio la resolucion de 768x1024 a 448x576

Impacto: cambia distribucion espacial, escala de detalles, latentes y memoria. Puede afectar textura, bordes y generalizacion a inferencia 768x1024.

### F-07. GarmentNet usa BF16 en V4/V5

El original usa FP32 en esa ruta. Los casts agregados en `ip_adapter/resampler.py` y `src/attentionhacked_tryon.py` evitan errores de dispositivo/dtype, pero el redondeo BF16 modifica features condicionantes.

Impacto: probablemente pequeño frente a resolucion/entrenamiento parcial, pero no bit-equivalente.

### F-08. Se desactivo el clipping efectivo de gradiente

El original aplica norma maxima 1.0. La version auditada usa `--max_grad_norm=0.0` por defecto, lo que lo desactiva.

Impacto: cambia cada update donde la norma exceda 1.0 y puede reducir estabilidad. La V6 debe medir normas antes de decidir una alternativa compatible con AMP.

### F-09. Los argumentos declarados no son los efectivos en low-VRAM

Con `--low_vram_training`:

- `mixed_precision=fp16` se convierte internamente en `no`.
- `weight_dtype` se fuerza a FP32.
- `--use_8bit_adam` termina usando `torch.optim.AdamW`.
- `--enable_xformers_memory_efficient_attention` se omite.

Impacto: logs/configuraciones pueden atribuir resultados a FP16, Adam8bit o xFormers cuando no se usaron. La interfaz V6 debe mostrar parametros solicitados y efectivos por separado.

### F-10. `seed=42` no inicializa el entrenamiento

La semilla se usa para imagenes de evaluacion, pero no se llama a un inicializador global antes del dataset/augmentaciones, ruido, timesteps o muestreo VAE. Ajustes deterministas de cuDNN no sustituyen el seed.

Impacto: corridas nominalmente iguales no son reproducibles y una comparacion A/B puede confundir variacion aleatoria con degradacion.

## Fallos heredados del original

### F-11. `time_ids` usa altura dos veces

`compute_time_ids` construye `target_size=(height, height)` y tambien recibe `(height, height)`. Para 448x576 o 768x1024 el condicionamiento SDXL declara una imagen cuadrada distinta de la real.

Impacto: condicionamiento espacial incorrecto. Corregirlo puede mejorar coherencia, pero cambia el modelo matematico y exige una rama experimental, no una correccion silenciosa.

### F-12. Rama `prediction_type == "sample"` usa `model_pred` sin definir

La variable producida es `noise_pred`; `model_pred = model_pred - noise` fallaria si el scheduler usa ese tipo. El scheduler actual suele usar `epsilon`, por lo que permanece latente.

Correccion: prueba unitaria de las tres ramas antes de cambiar; usar la variable correcta conforme al contrato del scheduler.

### F-13. Orden de datos fijo y estado de epoca ausente

`shuffle=False` y los bloques reinician desde el primer lote. En ejecuciones cortas repetidas, la cadena puede entrenar reiteradamente sobre el comienzo del dataset, aunque las augmentaciones sean aleatorias.

Impacto: sesgo fuerte de datos y comparacion engañosa de “steps acumulados”. Se requiere sampler reproducible y cursor persistido.

## Hallazgos medios y bajos

### F-14. Guardado de optimizer siempre consume RAM/disco

Aunque V4 evita cargar optimizer, `save_checkpoint` siempre construye y guarda su estado. Esto puede recrear el pico de RAM que se intentaba evitar.

Plan: serializacion medida y modo de warm-start sin promesa de continuidad; nunca omitir optimizer de una cadena `continue_exact`.

### F-15. Manifest insuficiente

No incluye hash de codigo, dataset, pesos base, estado RNG, versiones de librerias, GPU, padre, contador acumulado, checksum de archivos ni resultado de validacion. `completed_optimizer_updates` solo representa el bloque actual porque `global_step` vuelve a cero.

### F-16. `ejecutar_prueba_v4.sh` pierde su reporte en errores

Con `set -e`, si `accelerate launch` retorna no cero, el script puede salir antes de guardar `status=$?`, imprimir rutas y mostrar el final del log.

### F-17. Dos parsers distintos para `.env`

Shell hace `source .env`, lo cual ejecuta sintaxis shell. Gradio implementa un parser simple `KEY=VALUE`, no elimina comillas y no entiende `export`. Un mismo archivo puede tener valores efectivos distintos. `switch_model_version.sh` usa `sed` sin escapar todos los caracteres especiales del valor.

Plan: formato declarativo restringido, parser unico, escritura atomica y lista permitida de claves.

### F-18. `inference.sh` ejecuta cinco trabajos consecutivos

El script lanza VITON-HD paired/unpaired y tres categorias DressCode. Con `set -e`, cualquier fallo corta los restantes. No es una interfaz selectiva ni una prueba A/B controlada.

### F-19. Preprocesado CPU condicional

Los cambios en human parsing y OpenPose evitan `torch.cuda.set_device` cuando CUDA no esta disponible. Son operativos y no alteran resultados cuando CUDA si esta disponible; en CPU puede existir variacion de backend que debe registrarse.

### F-20. Traduccion de comentarios ASPP

Solo cambia documentacion interna; no tiene efecto de ejecucion.

## Cambios que pueden considerarse operativos

Son razonablemente neutros si sus pruebas confirman que no cambian tensores:

- Variables de ruta para seleccionar modelo en Gradio/inferencia.
- Guardas CPU antes de `torch.cuda.set_device`.
- `LD_LIBRARY_PATH`, asignador CUDA, locks, logs y chequeos de disco.
- Checkpoint atomico, siempre que el contenido sea identico al estado que sustituye.
- Menor numero de workers, si el orden, semillas por worker y batches son equivalentes.

No se debe extender esa etiqueta a resolucion, dtype, optimizer, clipping, parametros entrenables, estado de reanudacion o condicionamiento SDXL.
