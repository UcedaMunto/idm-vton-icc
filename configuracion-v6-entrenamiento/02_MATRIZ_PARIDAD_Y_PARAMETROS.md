# Matriz de paridad y parametros

## Clases de impacto

- `O`: operativo; no deberia cambiar el calculo del modelo.
- `N`: numerico; misma intencion, posible diferencia por precision/backend.
- `M`: matematico; cambia datos, objetivo, actualizacion o arquitectura entrenable.
- `R`: reanudacion/reproducibilidad.

| Parametro o cambio | Original | Perfil V4/V5 observado | Clase | Puede llamarse equivalente |
|---|---|---|---|---|
| Resolucion entrenamiento | 768x1024 por defecto | 448x576 | M | No |
| Batch | 6 por defecto | 1 | M | Solo si se compensa acumulacion y se prueba; hoy no |
| Acumulacion | 1 | 1 | M | Si, aislada; pero cambia batch efectivo junto con batch |
| Parametros entrenables | UNet completo | IP-Adapter/proyeccion/`conv_in` | M | No |
| Precision solicitada | segun Accelerate | `fp16` | N | No basta con el valor solicitado |
| Precision efectiva low-VRAM | no aplica | FP32 para ruta entrenable | N | No demostrada |
| GarmentNet | FP32 | BF16 | N | No bit-exacta |
| Optimizador solicitado | AdamW8bit posible | `--use_8bit_adam` | M | No basta con el valor solicitado |
| Optimizador efectivo low-VRAM | segun original | AdamW PyTorch | M | No |
| Clipping | norma 1.0 | desactivado por defecto | M | No |
| xFormers solicitado | opcional | activado en comando | N | No basta con el valor solicitado |
| xFormers efectivo low-VRAM | segun original | omitido | N | No demostrada |
| Gradient checkpointing | opcional | activo | N | Debe probar equivalencia de gradientes/tolerancia |
| Workers | 8/2 por defecto | 1/1 | O/R | Solo con orden y seeds por worker controlados |
| `pin_memory` | true | false low-VRAM | O | Probablemente, sujeto a prueba |
| Semilla | argumento 42 | no aplicada globalmente | R | No reproducible |
| Orden dataset | `shuffle=False` | `shuffle=False` | R/M | Igual al original, pero inadecuado para bloques cortos |
| Checkpoint | pipeline al final de epoca | compacto por update | O/R | Formato distinto; pesos pueden ser equivalentes |
| Optimizer al reanudar | no funcional en original | omitido por defecto | R/M | No |
| RNG/cursor al reanudar | no guardado | no guardado | R/M | No |
| Cast en Resampler/attention | no | conversion a dtype/device destino | N | No bit-exacta; evita fallos |
| Seleccion de modelo por env | modelo remoto fijo | ruta configurable | O | Si apunta al mismo artefacto |
| Guardas CUDA preprocesado | CUDA obligatoria | fallback CPU | O/N | Igual en CUDA; CPU debe medirse |

## Perfil solicitado frente a perfil efectivo

La V6 debe registrar ambos. Para el perfil actual:

```text
solicitado.mixed_precision = fp16
efectivo.mixed_precision = no
solicitado.optimizer = AdamW8bit
efectivo.optimizer = torch.optim.AdamW
solicitado.xformers = true
efectivo.xformers = false
efectivo.trainable_scope = ip_adapter + encoder_hid_proj + conv_in
efectivo.garmentnet_dtype = bfloat16
efectivo.resolution = 448x576
```

Si una interfaz muestra solo la columna solicitada, induce a conclusiones incorrectas.

## Perfiles V6 propuestos

### `original_reference`

Replica argumentos y comportamiento del original en una corrida de prueba separada. No se presupone que quepa en 12 GB; sirve como referencia documental y, si es necesario, se ejecuta en CPU/GPU mayor o con un subconjunto muy corto.

### `current_hardware_control`

Congela exactamente el estado auditado antes de corregir. Sirve para demostrar que correcciones operativas no cambian salidas mas alla de tolerancias.

### `v6_safe_candidate`

Incluye solo correcciones aprobadas una por una. Todo cambio `M` o `N` permanece visible y requiere experimento propio.

### `v6_quality_experiment`

Contiene cambios deliberados como corregir `time_ids`, restaurar clipping o cambiar resolucion. Nunca reemplaza al candidato seguro sin promocion explicita.

## Parametros bloqueados y editables en la interfaz

### Bloqueados por defecto

- Ruta de codigo y hash de commit.
- ID de linea y checkpoint padre.
- Dataset y hash de inventario.
- Alcance de parametros entrenables.
- Tipo de reanudacion.
- Resolucion del perfil aprobado.
- Scheduler/prediction type.

Cambiar uno crea una linea experimental nueva.

### Editables con validacion

- Numero de updates del bloque.
- Frecuencia de checkpoint dentro de limites.
- Workers 0-2 para el hardware actual.
- Umbral de disco/RAM.
- Logging y evaluacion.
- Ruta de salida nueva y vacia.

### Experimentales

- Dtype, optimizer, clipping, xFormers, resolucion, batch/acumulacion y alcance entrenable.

La interfaz debe exigir motivo, nueva linea y plan A/B cuando cambie cualquiera de estos.

## Invariantes

1. `width` y `height` son multiplos de 8 y coinciden con dataset, latentes, `time_ids` e inferencia de evaluacion.
2. `checkpointing_steps <= max_train_steps` en smoke tests.
3. `continue_exact` requiere optimizer, RNG, cursor, scaler y manifest compatibles.
4. `warm_start_weights` reinicia contadores y se marca como nueva fase.
5. La salida no puede ser el directorio del checkpoint padre.
6. Un checkpoint compacto no se ofrece como modelo de Gradio.
7. Un snapshot no se promociona sin evaluacion fija y checksums.
8. Parametros solicitados y efectivos deben coincidir o mostrar una advertencia bloqueante.
