# Plan de correcciones V6

Este plan no autoriza cambios de codigo ni ejecuciones largas. Cada fase exige evidencia y permite volver al ultimo artefacto validado.

## Fase 0. Congelar evidencia

1. Pausar cron/watchdog y confirmar que no se inicia una corrida nueva durante la auditoria.
2. No detener un entrenamiento activo sin decision explicita; registrar PID, comando, base, resume, salida y log.
3. Guardar `git status`, HEAD, diff no confirmado, hashes de scripts/configuraciones y versiones del entorno.
4. Crear un ID para cada linea historica: original, checkpoint-250, V3 compacta, V4 y futura V6.
5. Marcar V6 como `documentation_only` hasta cerrar las fases 1-4.

Aceptacion: se puede reconstruir que codigo, configuracion y padres produjeron cada artefacto relevante.

## Fase 1. Corregir automatizacion sin tocar matematicas

Verificado en vivo (2026-08-22): el watchdog actual YA implementa configuracion unica (`config_v4.env` via `source`) y un supervisor sincrono que espera el proceso, valida el checkpoint esperado y escribe `PAUSAR_POR_ERROR` de forma atomica ante fallo o exit code distinto de cero. Esta fase se reduce a cerrar las brechas remanentes, no a construir el supervisor desde cero.

### 1.1 Configuracion unica (parcialmente resuelto)

- Ya existe una unica fuente (`config_v4.env`) cargada por ruta explicita.
- Pendiente: migrar de `source` (shell ejecutable) a un formato declarativo no ejecutable para la futura interfaz, sin romper el watchdog shell actual mientras tanto.
- Pendiente: el monitor y el watchdog deben mostrar/registrar el hash del archivo de configuracion efectivo.

### 1.2 Supervisor real (ya implementado, reforzar validacion)

- Ya existe: `flock` para no duplicar el lanzador, deteccion de `train_xl.py` activo antes de lanzar, supervisor sincrono con `EXIT_CODE=$?`, validacion de `manifest.json` + `trainable_state.pt`, y disyuntor atomico.
- Pendiente: validar tambien `optimizer_state.pt` cuando corresponda, checksum de los archivos y compatibilidad de linaje/hash de codigo contra el manifest, no solo su existencia.
- Pendiente: registrar en el manifest un `parent_checkpoint_id` verificable para que el supervisor pueda rechazar un checkpoint de linea incompatible antes de promoverlo.

### 1.3 Pruebas negativas obligatorias

1. Comando invalido.
2. Checkpoint base ausente.
3. Proceso terminado con exit 1.
4. Proceso terminado por señal.
5. Exit 0 sin checkpoint.
6. Checkpoint parcial/sin manifest.
7. Disco bajo.
8. Lock ocupado.
9. Gradio activo.
10. Checkpoint incompatible.

Aceptacion: ninguna prueba produce un segundo lanzamiento; todas dejan diagnostico accionable.

## Fase 2. Definir semantica de checkpoint

### 2.1 `continue_exact`

Debe guardar y restaurar:

- Pesos entrenables y su esquema nombre/shape/dtype.
- Estado del optimizer y scaler.
- RNG Python, PyTorch CPU y todas las CUDA.
- Epoca, cursor/sampler y contador global acumulado.
- Parametros efectivos, hashes del codigo/dataset/base y versiones.
- Checksum y `parent_checkpoint_id`.

Si falta cualquier componente, aborta. No hay fallback silencioso.

### 2.2 `warm_start_weights`

Carga solo pesos, crea nuevo optimizer/RNG/cursor y abre una nueva fase. El manifest debe decir `continuity=false` y explicar el padre.

### 2.3 Snapshot de inferencia

Construir en una operacion separada desde un estado compacto validado. Ejecutar una carga limpia de Gradio/inferencia antes de promocionarlo.

Aceptacion: prueba continua de 4 updates y prueba 2+resume+2 cumplen tolerancia definida; la prueba falla si se elimina optimizer o RNG.

## Fase 3. Reproducibilidad y datos

1. Aplicar una semilla global antes de crear modelos/datasets.
2. Inicializar workers con semillas derivadas y registrar sus valores.
3. Usar sampler reproducible con cursor persistible.
4. Evitar que cada bloque corto reinicie siempre en la primera muestra.
5. Inventariar dataset por rutas, tamaños y hashes de metadatos/pares; no es necesario hashear imagenes en cada run si existe un inventario inmutable versionado.
6. Registrar augmentaciones efectivas.

Aceptacion: dos smoke tests desde el mismo estado producen indices, ruido/timesteps, perdida y pesos dentro de tolerancia.

## Fase 4. Corregir defectos de entrenamiento, uno por experimento

### 4.1 `time_ids`

Candidato: usar `(height, width)` tanto en `original_size` como en `target_size`. Comparar contra el control actual; no mezclar con dtype/resolucion.

### 4.2 Rama `sample`

Crear prueba unitaria para `epsilon`, `v_prediction` y `sample`. Corregir la variable conforme a Diffusers. El scheduler efectivo debe quedar en manifest.

### 4.3 Clipping

Medir norma de gradiente antes de elegir:

- referencia 1.0,
- desactivado actual,
- implementacion compatible con AMP.

No capturar y omitir silenciosamente el error de unscale; un fallo de clipping debe detener el experimento.

### 4.4 Estado optimizer y picos de RAM

Medir memoria durante construccion/serializacion. Si `continue_exact` no cabe, no llamarlo continuidad exacta: usar warm-start o serializacion por fragmentos probada.

Aceptacion: cada correccion tiene prueba aislada, diff pequeno, resultado A/B y decision mantener/rechazar.

## Fase 5. Separar hardware de calidad

Orden de experimentos:

1. Control actual reproducible.
2. Correcciones operativas de Fase 1.
3. Checkpoint/reanudacion exacta.
4. Seeds/sampler.
5. `time_ids`.
6. Clipping.
7. GarmentNet FP32 vs BF16.
8. Resolucion.
9. Alcance entrenable.
10. Optimizer.

No cambiar dos filas a la vez. Los cambios 7-10 son estrategias de entrenamiento, no arreglos neutrales.

## Fase 6. Interfaz

Implementar solo despues de estabilizar el esquema de configuracion. Primera version: editar, validar y generar; no ejecutar. Segunda version, opcional: smoke test manual con confirmacion y bloqueo.

Aceptacion: configuraciones invalidas no se guardan; cada guardado crea revision, diff y comando efectivo; nunca sobrescribe una salida existente.

## Fase 7. Promocion gradual

1. Validacion estatica y tests unitarios.
2. Smoke sin entrenamiento o carga de modelos.
3. Un update.
4. Reanudacion 1+1.
5. Diez updates.
6. Comparacion de calidad fija.
7. Bloque de 100 solo si todo lo anterior aprueba.
8. Cron solo despues de una prueba de fallo supervisada.

## Retroceso

- Los checkpoints historicos son inmutables.
- Cada candidato escribe en un directorio nuevo.
- La configuracion activa se cambia por un puntero/revision, no editando artefactos.
- Ante NaN, OOM, fallo de integridad, regresion visual o incompatibilidad, se pausa y vuelve al ultimo snapshot promovido.
- Nunca se borra el unico estado recuperable ni el control usado en una comparacion.

## Criterio de cierre

V6 puede declararse lista cuando:

- No quedan hallazgos bloqueantes/altos sin decision documentada.
- Watchdog y monitor muestran la misma configuracion efectiva.
- `continue_exact` supera la prueba continua contra reanudada.
- Existe snapshot de inferencia enlazado y cargable.
- La comparacion A/B no muestra degradacion segun criterios predefinidos.
- La interfaz no permite combinaciones incompatibles ni inicio accidental.
