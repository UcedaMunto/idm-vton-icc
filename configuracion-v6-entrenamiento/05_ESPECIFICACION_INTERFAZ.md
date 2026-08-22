# Especificacion de interfaz de configuracion V6

## Objetivo

Una interfaz local pequena para crear configuraciones revisables sin editar shell manualmente. La primera entrega solo valida, guarda revisiones y genera comandos; no inicia entrenamientos.

Tecnologia propuesta: Gradio ya existe en el proyecto y evita introducir otro framework. La logica de esquema/validacion debe vivir en un modulo independiente de Gradio para poder probarla sin UI.

## Principios

- Perfil y parametros efectivos visibles antes de guardar.
- Defaults conservadores provenientes de una revision validada, no hardcodeados en varios scripts.
- Ningun campo libre termina interpolado en shell.
- Argumentos se construyen como lista y se ejecutarian sin `shell=True` en una fase futura.
- Las rutas se normalizan y deben estar dentro de raices permitidas.
- Guardado atomico, historial inmutable y diff entre revisiones.
- Cambios matematicos abren una linea experimental y muestran advertencia bloqueante.
- Guardar nunca inicia una corrida.

## Pantallas

### 1. Estado

- Configuracion activa: ruta, revision y hash.
- Codigo: commit, indicador dirty y hash del diff.
- GPU/RAM/disco.
- Proceso de entrenamiento/Gradio y archivos de pausa.
- Ultimo checkpoint compacto y ultimo snapshot promovido.
- Discrepancias entre solicitado y efectivo.

Solo lectura.

### 2. Perfil

Selector:

- `current_hardware_control`
- `v6_safe_candidate`
- `v6_quality_experiment`
- `original_reference` solo lectura por defecto

Muestra padre, alcance entrenable, resolucion, dtype, optimizer y tipo de continuidad.

### 3. Entrenamiento

Controles adecuados:

- Stepper: updates del bloque y checkpoint interval.
- Selectores: precision, optimizer, trainable scope y modo de reanudacion.
- Campos numericos: learning rate, clipping, batch, acumulacion y workers.
- Toggles: gradient checkpointing, xFormers y snapshot de inferencia.
- Rutas con selector/validacion: dataset, base, resume y salida.

Los parametros experimentales aparecen agrupados con impacto `M/N/R`, no escondidos.

### 4. Seguridad

- Umbral de disco libre.
- Umbral de RAM y swap.
- Exclusividad GPU con Gradio.
- Duracion maxima del smoke.
- Politica de retencion en modo simulacion.
- Toggle de cron deshabilitado hasta que exista evidencia de supervisor aprobado.

### 5. Revision

Antes de guardar muestra:

- Diff contra revision padre.
- Parametros solicitados frente a efectivos.
- Cambios clasificados `O/N/M/R`.
- Validaciones y advertencias.
- Comando como lista escapada solo para inspeccion.
- Directorios/archivos que se crearian.
- ID de linea y corrida propuestos.

Botones: `Validar`, `Guardar revision`, `Exportar informe`. No incluir `Entrenar` en V6.0.

### 6. Comparacion

Formulario para seleccionar control/candidato y conjunto fijo; genera el plan de ejecucion y la hoja de evaluacion. No permite comparar si cambian variables adicionales no declaradas.

## Esquema conceptual

```yaml
schema_version: 1
revision_id: generated
parent_revision_id: required
mode: warm_start_weights | continue_exact
line_id: required
paths:
  base_checkpoint: required
  resume_checkpoint: optional
  data_dir: required
  output_root: required-new
model:
  width: 448
  height: 576
  trainable_scope: ip_adapter
  garmentnet_dtype: bfloat16
optimization:
  optimizer: adamw
  learning_rate: 1.0e-5
  max_grad_norm: 0.0
  batch_size: 1
  gradient_accumulation_steps: 1
runtime:
  mixed_precision_requested: fp16
  low_vram: true
  gradient_checkpointing: true
  xformers_requested: true
  train_workers: 1
checkpoint:
  block_updates: 1
  every_updates: 1
  full_snapshot: false
safety:
  minimum_free_gib: 90
  deny_when_gradio_active: true
```

El archivo real puede ser JSON o YAML, pero debe leerse con parser estructurado. No usar `.env` ejecutable como formato canonico.

## Validaciones bloqueantes

1. Base completa contiene componentes requeridos y checksums/manifest aceptados.
2. Compacto coincide con line ID, padre, esquema de tensores y parametros efectivos.
3. `continue_exact` tiene optimizer, RNG, cursor y scaler cuando corresponda.
4. Resolucion positiva, multiplo de 8 y coherente con `time_ids`.
5. Batch efectivo y acumulacion declarados.
6. Salida nueva, no padre ni subdirectorio de un artefacto inmutable.
7. Espacio suficiente para estado, snapshot y reserva.
8. No hay Gradio/entrenamiento activo.
9. No hay dirty code sin hash del diff y autorizacion experimental.
10. Flags incompatibles no se aceptan; no se corrigen silenciosamente.
11. Un parametro solicitado que el runtime ignorara debe bloquear o requerir aceptacion explicita.
12. No se puede activar cron desde una revision no promovida.

## Validaciones de advertencia

- BF16 frente a FP32 de referencia.
- Resolucion distinta a inferencia/evaluacion.
- Warm-start sin optimizer.
- Clipping desactivado.
- Dataset sin inventario.
- Menos de 30 muestras en comparacion de promocion.

## Persistencia

```text
configuracion-v6-entrenamiento/revisiones/<revision_id>/config.yaml
configuracion-v6-entrenamiento/revisiones/<revision_id>/effective.json
configuracion-v6-entrenamiento/revisiones/<revision_id>/validation.json
configuracion-v6-entrenamiento/revisiones/<revision_id>/diff.md
```

No crear estas rutas hasta implementar y probar el esquema. Nunca guardar secretos.

## Pruebas de la interfaz

- Unitarias de esquema y reglas cruzadas.
- Paths con espacios, comillas, `&`, `|`, saltos y traversal.
- Config corrupta/clave desconocida/version futura.
- Carrera entre dos guardados.
- Directorio de salida existente.
- Checkpoint parcial/incompatible.
- Vista solicitada/efectiva para low-VRAM.
- Confirmar que `Guardar revision` no crea procesos.
- Capturas desktop/movil sin solapamientos si se publica por navegador.

## Evolucion opcional V6.1

Solo despues de aprobar supervisor y smoke tests puede agregarse `Ejecutar smoke`. Debe requerir confirmacion final, lock, salida nueva, limite de un update y enlace al log. Cron sigue siendo una operacion administrativa separada.
