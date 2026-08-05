# Guia de Ajustes IDM-VTON para RTX 3060 12GB

Fecha de validacion: 2026-06-21
Proyecto: IDM-VTON

## Resumen ejecutivo

Este documento resume los cambios que permitieron estabilizar el proyecto en una GPU de 12 GB (RTX 3060), evitando errores de memoria y errores de offload.

Estado actual validado:
- La aplicacion esta levantada en puerto 7860.
- El proceso de la app usa GPU activamente.
- El pipeline funciona en modo hibrido: GPU + offload a CPU (no modo CPU puro).

Evidencia tecnica de estado actual:
- Listener activo: 0.0.0.0:7860
- Proceso de compute en GPU: PID 9679 (python) con memoria asignada en GPU

### Archivos de configuracion modificados en esta seccion
- `gradio_demo/app.py`
  - Configura selecccion de dispositivo (CUDA/CPU), precision (`float16` en GPU), perfil low-ram y offload para reducir consumo de VRAM.
- `run_gpu_with_logs.sh`
  - Define variables de entorno por defecto para ejecucion estable en 12 GB y arranque reproducible con logs.

## Causa raiz de los fallos observados

1. OOM en Try-on
- El pipeline SDXL con modulos extra puede superar 12 GB durante picos de inferencia.
- En algunos intentos, la memoria quedaba muy alta entre requests y el siguiente intento iniciaba sin margen.

2. Error Cannot copy out of meta tensor; no data!
- Ocurria al mezclar CPU offload (accelerate) con movimientos manuales de tensores/modelos mediante .to(...).
- Con offload activo, algunos componentes quedan como meta tensors y no se deben copiar manualmente.

### Archivos de configuracion modificados en esta seccion
- Ninguno en esta seccion (analisis diagnostico).

## Ajustes aplicados en gradio_demo/app.py

### 1) Seleccion de dispositivo y modos de ejecucion
- Variables de entorno agregadas/normalizadas:
  - IDMVTON_FORCE_CPU
  - IDMVTON_CPU_OFFLOAD
  - IDMVTON_LOW_RAM
- Logica:
  - USE_CUDA = torch.cuda.is_available() y no FORCE_CPU
  - dtype en GPU: float16

**Archivo de configuracion modificado:**
- `gradio_demo/app.py`

**Que hace el cambio:**
- `IDMVTON_FORCE_CPU`: fuerza modo CPU cuando vale `1`.
- `IDMVTON_CPU_OFFLOAD`: habilita offload de modulos a CPU para bajar pico de VRAM.
- `IDMVTON_LOW_RAM`: activa perfil reducido para GPUs de 12 GB.
- `USE_CUDA` y `torch_dtype`: ejecutan en GPU con `float16` cuando hay CUDA disponible y no se fuerza CPU.

### 2) Perfil low-ram para 12 GB
- Resolucion interna en LOW_RAM_MODE:
  - TARGET_WIDTH = 576
  - TARGET_HEIGHT = 768
- En modo normal:
  - 768 x 1024

**Archivo de configuracion modificado:**
- `gradio_demo/app.py`

**Que hace el cambio:**
- Ajusta resolucion objetivo (`TARGET_WIDTH`, `TARGET_HEIGHT`) a `576x768` en low-ram para reducir consumo de memoria durante preprocesamiento e inferencia.

### 3) Reduccion de pico de VRAM en pipeline
- attention slicing activado (modo agresivo): enable_attention_slicing(1)
- VAE tiling activado (si esta disponible): enable_vae_tiling()
- VAE slicing activado (si esta disponible): enable_vae_slicing()
- xformers memory efficient attention: intento de activacion con fallback silencioso

**Archivo de configuracion modificado:**
- `gradio_demo/app.py`

**Que hace el cambio:**
- `enable_attention_slicing(1)`: procesa atencion por partes para bajar picos de VRAM.
- `enable_vae_tiling()` y `enable_vae_slicing()`: reducen memoria en decodificacion VAE.
- `enable_xformers_memory_efficient_attention()`: usa atencion eficiente si esta instalada; si no, mantiene fallback seguro.

### 4) Offload a CPU para estabilidad en 12 GB
- Si IDMVTON_CPU_OFFLOAD=1:
  - Se habilita enable_sequential_cpu_offload()
  - Fallback a enable_model_cpu_offload() si el anterior no aplica
- Importante:
  - Se evita mover manualmente partes del pipeline con .to(...) cuando offload esta activo.
  - Esto evita el error de meta tensor.

**Archivo de configuracion modificado:**
- `gradio_demo/app.py`

**Que hace el cambio:**
- Activa `enable_sequential_cpu_offload()` como primera opcion.
- Usa `enable_model_cpu_offload()` como fallback.
- Evita llamadas manuales a `.to(...)` sobre el pipeline cuando offload esta activo para prevenir errores con meta tensors.

### 5) DensePose y OpenPose para reducir carga de GPU
- DensePose en CPU cuando LOW_RAM_MODE y USE_CUDA estan activos.
- OpenPose se mueve temporalmente a GPU para inferencia de mask y luego regresa a CPU.

**Archivo de configuracion modificado:**
- `gradio_demo/app.py`

**Que hace el cambio:**
- Define `densepose_device='cpu'` en low-ram para liberar VRAM para SDXL.
- Mueve OpenPose solo durante el calculo de mascara y lo devuelve a CPU para estabilizar uso de memoria.

### 6) Limpieza de memoria robusta
- Manejo de excepciones OOM (torch.cuda.OutOfMemoryError y RuntimeError con out of memory).
- Limpieza con torch.cuda.empty_cache().
- Bloque finally para limpieza consistente.
- Regla critica: no hacer pipe.to("cpu") en modo CPU offload para evitar meta tensor error.

**Archivo de configuracion modificado:**
- `gradio_demo/app.py`

**Que hace el cambio:**
- Captura `torch.cuda.OutOfMemoryError` y errores runtime de tipo OOM.
- Ejecuta `torch.cuda.empty_cache()` en errores y en `finally`.
- En `finally`, evita `pipe.to("cpu")` si `CPU_OFFLOAD=1` para no disparar el error de meta tensor.

### 7) Parametros de UI para bajar consumo
- Denoising Steps por defecto ajustado a 16
- Rango permitido: 12 a 40

**Archivo de configuracion modificado:**
- `gradio_demo/app.py`

**Que hace el cambio:**
- Ajusta el control de UI de `Denoising Steps` para mantener configuraciones de inferencia con menor consumo de VRAM por defecto.

## Ajustes aplicados en run_gpu_with_logs.sh

### 1) Arranque robusto y reproducible
- Verificacion de puerto 7860 antes de arrancar
- Verificacion de conda
- Logs por timestamp + symlink latest.log

**Archivo de configuracion modificado:**
- `run_gpu_with_logs.sh`

**Que hace el cambio:**
- Valida existencia de `gradio_demo/app.py`.
- Verifica que el puerto `7860` este libre antes de iniciar.
- Detecta `conda` (binario normal o ruta fija) y corta con error claro si no existe.
- Crea logs versionados y actualiza `logs/latest.log` para monitoreo rapido.

### 2) Variables por defecto orientadas a 12 GB
- IDMVTON_LOW_RAM=1
- IDMVTON_CPU_OFFLOAD=1
- IDMVTON_FORCE_CPU=0
- PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

**Archivo de configuracion modificado:**
- `run_gpu_with_logs.sh`

**Que hace el cambio:**
- Exporta defaults de bajo consumo para memoria en GPUs de 12 GB.
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64` reduce fragmentacion y mejora estabilidad ante picos de asignacion.

### 3) Ejecucion en entorno correcto
- Siempre usa conda run -n idm python gradio_demo/app.py

**Archivo de configuracion modificado:**
- `run_gpu_with_logs.sh`

**Que hace el cambio:**
- Fuerza ejecucion dentro del entorno `idm` para evitar errores por usar Python/paquetes del entorno base.

## Confirmacion: Se esta usando GPU o CPU?

Respuesta corta:
- Se esta usando GPU, con offload parcial a CPU.

Respuesta tecnica:
- El backend corre en CUDA cuando IDMVTON_FORCE_CPU=0 y hay GPU disponible.
- Offload mueve partes del pipeline a CPU para ahorrar VRAM, pero la inferencia principal sigue aprovechando GPU.

Como validar en cualquier momento:
1. Ver puerto:
   - ss -ltnp | grep :7860
2. Ver proceso Python en GPU:
   - nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
3. Ver estado web:
   - curl -I -s http://127.0.0.1:7860 | head -n 1

### Archivos de configuracion modificados en esta seccion
- Ninguno en esta seccion (solo validacion operativa del estado en runtime).

## Comandos operativos recomendados

Arranque estable:
- cd /home/uceda/Documents/IDM-VTON && pkill -f 'python .*gradio_demo/app.py' || true && nohup /home/uceda/Documents/IDM-VTON/run_gpu_with_logs.sh > /home/uceda/Documents/IDM-VTON/logs/launcher_nohup.out 2>&1 &

Monitoreo de logs:
- tail -f /home/uceda/Documents/IDM-VTON/logs/latest.log

### Archivos de configuracion modificados en esta seccion
- Ninguno adicional (se utilizan los cambios ya aplicados en `run_gpu_with_logs.sh`).

## Perfil sugerido para RTX 3060 12GB

Para mayor estabilidad:
- Mantener Denoising Steps entre 12 y 20 (ideal 16)
- Procesar una solicitud Try-on a la vez
- Evitar otras cargas pesadas de GPU en paralelo

Fallback de emergencia (CPU puro):
- IDMVTON_FORCE_CPU=1 /home/uceda/Documents/IDM-VTON/run_gpu_with_logs.sh

### Archivos de configuracion modificados en esta seccion
- Ninguno en esta seccion (recomendaciones de uso sobre configuraciones ya introducidas).

## Archivos modificados (fuente de verdad)

- gradio_demo/app.py
- run_gpu_with_logs.sh

### Que hace cada cambio de configuracion (resumen)
- `gradio_demo/app.py`
  - Gestiona modo CUDA/CPU, low-ram, offload, optimizaciones de memoria (attention/VAE/xformers), control de OOM y defaults de UI.
- `run_gpu_with_logs.sh`
  - Estandariza variables de entorno, valida precondiciones de arranque y garantiza ejecucion en entorno `conda` correcto con trazabilidad en logs.

## Notas finales

- Exit code 143 durante reinicios es normal cuando se termina una instancia previa.
- ModuleNotFoundError: gradio aparece solo si se ejecuta con python del entorno base, no con conda env idm.

### Archivos de configuracion modificados en esta seccion
- Ninguno en esta seccion (notas de operacion y diagnostico).
