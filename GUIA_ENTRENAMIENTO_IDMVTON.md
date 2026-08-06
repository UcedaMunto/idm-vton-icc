# Guia Minuciosa de Entrenamiento IDM-VTON (VITON-HD)

Fecha: 2026-08-02
Proyecto: IDM-VTON

## 1) Estado actual del sistema (validado)

Validacion realizada en este servidor:
- Servicio Gradio activo en `http://127.0.0.1:7860/`
- Puerto escuchando: `0.0.0.0:7860`
- Respuesta HTTP: `200 OK`
- Proceso Python en GPU: PID `39448`
- Memoria GPU reportada por `nvidia-smi`: `106 MB` al momento de validacion

Interpretacion:
- El entorno y la app estan funcionando correctamente.
- Para entrenar, conviene detener temporalmente la demo para liberar GPU/VRAM.

Comando recomendado antes de entrenar:

```bash
cd /home/uceda/Documents/IDM-VTON
pkill -f 'python .*gradio_demo/app.py' || true
```

---

## 2) Alcance y supuestos de esta guia

Esta guia asume que:
- Ya tienes la bateria de entrenamiento lista (dataset preparado).
- Ya tienes el entorno conda funcional.
- Ya tienes checkpoints auxiliares y dependencias instaladas.

### Como verificar los 3 supuestos antes de entrenar

#### 1. Verificar el entorno

Ejecuta esto para confirmar que estas en el entorno correcto y que las librerias base responden:

```bash
cd /home/uceda/Documents/IDM-VTON
source ~/miniconda3/etc/profile.d/conda.sh
conda activate idm
which python
python -c "import torch, accelerate, diffusers, transformers; print(torch.__version__); print(torch.cuda.is_available())"
```

Debes comprobar:
- Que `which python` apunte al entorno `idm`.
- Que `torch.cuda.is_available()` devuelva `True` si vas a entrenar con GPU.
- Que no haya errores de import al cargar `torch`, `accelerate`, `diffusers` o `transformers`.

#### 2. Verificar la bateria de entrenamiento

Ejecuta esto para confirmar que el dataset esta completo y con la estructura esperada:

```bash
cd /ruta/a/tu/DATA_DIR
find train -maxdepth 2 -type f | sort | head -n 50
find test -maxdepth 2 -type f | sort | head -n 50
test -f train/vitonhd_train_tagged.json && echo "train json OK"
test -f test/vitonhd_test_tagged.json && echo "test json OK"
test -f train_pairs.txt && echo "train pairs OK"
test -f test_pairs.txt && echo "test pairs OK"
```

Debes comprobar:
- Que existan `train/` y `test/`.
- Que existan `image/`, `image-densepose/`, `agnostic-mask/` y `cloth/` dentro de cada particion.
- Que existan los archivos `vitonhd_train_tagged.json`, `vitonhd_test_tagged.json`, `train_pairs.txt` y `test_pairs.txt`.

#### 3. Verificar los checkpoints

Ejecuta esto para confirmar que los pesos auxiliares estan presentes:

```bash
cd /home/uceda/Documents/IDM-VTON
test -f ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin && echo "ip-adapter OK"
test -f ckpt/image_encoder/config.json && echo "image_encoder config OK"
test -f ckpt/image_encoder/model.safetensors && echo "image_encoder weights OK"
ls -lh ckpt/densepose/model_final_162be9.pkl
ls -lh ckpt/humanparsing/parsing_atr.onnx
ls -lh ckpt/humanparsing/parsing_lip.onnx
ls -lh ckpt/openpose/ckpts/body_pose_model.pth
```

Debes comprobar:
- Que el IP-Adapter exista en `ckpt/ip_adapter/`.
- Que el image encoder exista en `ckpt/image_encoder/`.
- Que DensePose, Human Parsing y OpenPose esten descargados en `ckpt/`.

Esta guia cubre:
- Entrenamiento sobre VITON-HD con `train_xl.py`.
- Verificacion previa, arranque, monitoreo, checkpoints y troubleshooting.

---

## 3) Estructura esperada del dataset (obligatoria)

El script `train_xl.py` espera esta estructura en `--data_dir`:

```text
DATA_DIR/
  train/
    image/
    image-densepose/
    agnostic-mask/
    cloth/
    vitonhd_train_tagged.json
  test/
    image/
    image-densepose/
    agnostic-mask/
    cloth/
    vitonhd_test_tagged.json
  train_pairs.txt
  test_pairs.txt
```

Detalles importantes del parser de dataset:
- El loader abre:
  - `train/vitonhd_train_tagged.json`
  - `test/vitonhd_test_tagged.json`
- Lee pares desde:
  - `train_pairs.txt`
  - `test_pairs.txt`
- Las mascaras se esperan como:
  - `agnostic-mask/<nombre>_mask.png`
- Las imagenes DensePose se esperan en:
  - `image-densepose/<nombre>.jpg` (o extension equivalente usada en tus pares)

Si algo no coincide con nombres/rutas, el entrenamiento falla por archivo faltante.

---

## 4) Archivos clave del entrenamiento en este repo

- `train_xl.py`: script principal de entrenamiento.
- `train_xl.sh`: ejemplo rapido de lanzamiento multi-GPU.
- `environment.yaml`: versiones recomendadas base.
- `ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin`: pesos IP-Adapter.
- `ckpt/image_encoder/`: encoder de imagen (CLIP) para IP-Adapter.

Validado en este workspace:
- `ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin` existe.
- `ckpt/image_encoder/config.json` existe.
- `ckpt/image_encoder/model.safetensors` existe.

---

## 5) Parametros de entrenamiento mas importantes

Parametros CLI relevantes de `train_xl.py`:
- `--data_dir`: ruta al dataset.
- `--output_dir`: carpeta de salida.
- `--train_batch_size`: batch por dispositivo.
- `--test_batch_size`: batch de validacion visual.
- `--num_train_epochs`: epocas objetivo (default 130).
- `--max_train_steps`: si se define, pisa epocas.
- `--learning_rate`: default `1e-5`.
- `--gradient_checkpointing`: reduce VRAM (mas lento).
- `--use_8bit_adam`: reduce consumo de optimizador (bitsandbytes).
- `--mixed_precision`: `fp16` recomendado en RTX 30xx.
- `--enable_xformers_memory_efficient_attention`: reduce memoria si xformers esta instalado.
- `--logging_steps`: cada cuantos pasos hace validacion/guardado de imagen de test.
- `--checkpointing_epoch`: frecuencia de guardado de checkpoints (nota en seccion 11).

Resolucion:
- `--width` default `768`
- `--height` default `1024`

---

## 6) Preparacion recomendada para entrenar en RTX 3060 12GB

Perfil recomendado (estable):
- `mixed_precision=fp16`
- `gradient_checkpointing` activado
- `use_8bit_adam` activado
- `train_batch_size=1` o `2` por GPU
- Reducir workers si hay presion de RAM de sistema

Sugerencia de variables para reducir fragmentacion CUDA:

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
```

Nota:
- `IDMVTON_LOW_RAM`, `IDMVTON_CPU_OFFLOAD`, `IDMVTON_FORCE_CPU` son para demo/inferencia en `gradio_demo/app.py`.
- `train_xl.py` no usa esas variables para la logica principal de entrenamiento.

---

## 7) Comandos de arranque (paso a paso)

### 7.1 Activar entorno

```bash
cd /home/uceda/Documents/IDM-VTON
source ~/miniconda3/etc/profile.d/conda.sh
conda activate idm
```

### 7.2 Comprobar GPU y librerias

```bash
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import accelerate, diffusers, transformers; print('ok')"
```

### 7.3 (Opcional) Configurar Accelerate

```bash
accelerate config
```

Recomendado en una sola GPU:
- `num_processes=1`
- `mixed_precision=fp16`

### 7.4 Secuencia completa (copiar y ejecutar en orden)

```bash
# 0) Entrar al proyecto y activar entorno
cd /home/uceda/Documents/IDM-VTON
source ~/miniconda3/etc/profile.d/conda.sh
conda activate idm

# 1) Definir rutas principales (OBLIGATORIO ajustar DATA_DIR)
export DATA_DIR=/ruta/real/a/tu/dataset/VITON-HD/zalando
export OUTPUT_DIR=/home/uceda/Documents/IDM-VTON/result_train

# 2) Definir parámetros de entrenamiento
export TRAIN_BATCH_SIZE=1
export TEST_BATCH_SIZE=1
export NUM_TRAIN_EPOCHS=130
export LOGGING_STEPS=1000
export CHECKPOINT_EPOCH=10
export MIXED_PRECISION=fp16
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

# 3) Verificaciones rápidas antes de lanzar
echo "DATA_DIR=$DATA_DIR"
test -d "$DATA_DIR" || { echo "ERROR: DATA_DIR no existe"; return 1; }
test -f "$DATA_DIR/train/vitonhd_train_tagged.json" || { echo "ERROR: falta train/vitonhd_train_tagged.json"; return 1; }
test -f "$DATA_DIR/test/vitonhd_test_tagged.json" || { echo "ERROR: falta test/vitonhd_test_tagged.json"; return 1; }
which accelerate || { echo "ERROR: accelerate no está en PATH"; return 1; }

# 4) Recuperación de caché SDXL corrupta (seguro si ya está bien)
rm -rf ~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0

# 5) Lanzar entrenamiento
accelerate launch train_xl.py \
  --gradient_checkpointing \
  --use_8bit_adam \
  --mixed_precision=${MIXED_PRECISION} \
  --train_batch_size=${TRAIN_BATCH_SIZE} \
  --test_batch_size=${TEST_BATCH_SIZE} \
  --num_train_epochs=${NUM_TRAIN_EPOCHS} \
  --logging_steps=${LOGGING_STEPS} \
  --checkpointing_epoch=${CHECKPOINT_EPOCH} \
  --output_dir=${OUTPUT_DIR} \
  --data_dir=${DATA_DIR}
```

Nota:
- Si ejecutas en shell no interactiva y falla `return 1`, reemplázalo por `exit 1`.
- `DATA_DIR` no puede quedar vacío; debe apuntar a la carpeta del dataset con `train/` y `test/`.

---

## 8) Monitoreo durante entrenamiento

### GPU/VRAM en vivo

```bash
watch -n 1 nvidia-smi
```

### Procesos Python de entrenamiento

```bash
ps -fp $(pgrep -f 'train_xl.py' | tr '\n' ' ')
```

### Ver resultados generados en validacion

El script guarda muestras de test en `output_dir` con patron:
- `<global_step>_<idx>_test.jpg`

Ejemplo:

```bash
ls -lh /home/uceda/Documents/IDM-VTON/result_train | head
```

---

## 9) Salidas esperadas

Dentro de `--output_dir` deberias ver:
- Imagenes de test periodicas (`*_test.jpg`)
- Checkpoints tipo:
  - `checkpoint-<global_step>/`

Cada checkpoint guardado contiene pipeline serializado para reutilizar.

---

## 10) Reanudar o continuar entrenamiento

El script no expone explicitamente `--resume_from_checkpoint` en los argumentos.

Estrategia practica de continuidad:
1. Usar un checkpoint guardado como nuevo `--pretrained_model_name_or_path`.
2. Mantener coherentes los argumentos de resolucion, tokenizer y componentes auxiliares.

Ejemplo de continuidad (adaptar ruta):

```bash
accelerate launch train_xl.py \
  --pretrained_model_name_or_path=/home/uceda/Documents/IDM-VTON/result_train/checkpoint-XXXX \
  --pretrained_garmentnet_path=stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_ip_adapter_path=ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin \
  --image_encoder_path=ckpt/image_encoder \
  --gradient_checkpointing --use_8bit_adam --mixed_precision=fp16 \
  --train_batch_size=1 --test_batch_size=1 \
  --output_dir=/home/uceda/Documents/IDM-VTON/result_train_continue \
  --data_dir=DATA_DIR
```

---

## 11) Comportamientos del script que debes conocer

1. `checkpointing_epoch` se evalua contra `global_step` al final de cada epoca.
- En la practica, el guardado ocurre cuando al cerrar una epoca se cumple `global_step % checkpointing_epoch == 0`.

2. `num_workers` en DataLoader:
- Train usa `num_workers=16`.
- Test usa `num_workers=4`.

Si tienes limites de RAM/CPU, considera bajar workers editando script en una rama de trabajo.

3. El entrenamiento incluye validacion visual periodica:
- En cada `logging_steps` crea pipeline temporal y genera una muestra de test.
- Esto puede introducir picos de VRAM/tiempo.

---

## 12) Troubleshooting rapido

### Error OOM (CUDA out of memory)
Acciones en orden:
1. Bajar `--train_batch_size` a `1`.
2. Asegurar `--gradient_checkpointing` y `--use_8bit_adam`.
3. Mantener `--mixed_precision=fp16`.
4. Cerrar cualquier proceso GPU paralelo (incluida demo Gradio).

### Error de bitsandbytes
- Verifica instalacion en entorno `idm`:

```bash
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
```

Si falla, reinstala `bitsandbytes` segun tu stack CUDA.

### Cuello de botella de disco/CPU
- Si carga de datos es lenta, reducir augmentations o ajustar workers puede estabilizar.

---

## 13) Flujo recomendado de trabajo

1. Detener demo Gradio.
2. Verificar entorno y GPU.
3. Lanzar entrenamiento con batch conservador.
4. Monitorear VRAM y muestras de test.
5. Guardar checkpoints y evaluar calidad.
6. Reabrir demo cuando no estes entrenando.

Para relanzar demo luego de entrenar:

```bash
cd /home/uceda/Documents/IDM-VTON
nohup /home/uceda/Documents/IDM-VTON/run_gpu_with_logs.sh > /home/uceda/Documents/IDM-VTON/logs/launcher_nohup.out 2>&1 &
```

---

## 14) Checklist final antes de pulsar Enter

- [ ] `DATA_DIR` correcto y completo.
- [ ] `ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin` presente.
- [ ] `ckpt/image_encoder/` presente.
- [ ] Demo Gradio detenida para liberar VRAM.
- [ ] `train_batch_size` ajustado a tu VRAM real.
- [ ] `output_dir` en disco con espacio suficiente.

Con esto, tienes un flujo de entrenamiento completo, reproducible y adaptado al estado actual estable de tu servidor.
