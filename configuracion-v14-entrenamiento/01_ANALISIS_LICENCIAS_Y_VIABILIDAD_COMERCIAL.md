# 01 - Analisis de licencias y viabilidad comercial

Fecha: 2026-09-09. Todas las licencias de esta tabla se verificaron contra el
texto oficial (archivo de licencia del repositorio o README del autor). Las
fuentes exactas estan al final.

## Cadena completa del proyecto (que se usa y bajo que licencia)

| # | Componente | Ruta / identificador local | Rol en el proyecto | Licencia | Comercial |
|---|---|---|---|---|:--:|
| 1 | Codigo IDM-VTON | `src/`, `train_xl.py`, `inference.py`, `gradio_demo/` | Framework try-on + entrenamiento | CC BY-NC-SA 4.0 | NO |
| 2 | Checkpoints IDM-VTON | `result_train_v14/base_oficial` -> cache `models--yisol--IDM-VTON` | Base de inferencia y del fine-tune | CC BY-NC-SA 4.0 | NO |
| 3 | Dataset VITON-HD | `dataset/DATA_DIR_PREP` (`train/`, `test/`) | Datos de entrenamiento/validacion | CC BY-NC 4.0 "research purposes only" | NO |
| 4 | DensePose | `ckpt/densepose/model_final_162be9.pkl` | Representacion densa de pose (input) | CC BY-NC 4.0 | NO |
| 5 | OpenPose | `ckpt/openpose/ckpts/body_pose_model.pth` | Pose (generacion de mascara) | Solo uso no comercial (comercial via FlintBox) | NO |
| 6 | SCHP (human parsing) | `ckpt/humanparsing/parsing_{atr,lip}.onnx` | Mascara agnostica de prenda | MIT (codigo) | SI (matices) |
| 7 | IP-Adapter | `ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin`, `IP-Adapter/` | Conditioning visual de la prenda | Apache-2.0 | SI |
| 8 | SDXL base 1.0 | `stabilityai/stable-diffusion-xl-base-1.0` (GarmentNet) | Encoder de prenda | CreativeML OpenRAIL++-M | SI (restricciones) |
| 9 | SDXL inpainting 0.1 | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | Base de difusion | CreativeML OpenRAIL++-M | SI (restricciones) |

## Veredicto

1. **Bloqueadores duros (NO comerciales):** #1 y #2 (IDM-VTON, CC BY-NC-SA 4.0),
   #3 (VITON-HD, CC BY-NC 4.0) y #4/#5 (DensePose / OpenPose, solo no comercial).
   Basta **uno** de ellos para impedir el uso comercial.
2. **Obra derivada:** un fine-tune de `yisol/IDM-VTON` entrenado con VITON-HD es
   una obra derivada de ambos. Hereda la restriccion **NC** y, por la clausula
   **SA (ShareAlike)**, cualquier distribucion derivada debe quedar tambien bajo
   CC BY-NC-SA 4.0. No se puede "relicenciar" a comercial.
3. **Basta con tener piezas permisivas:** IP-Adapter (Apache-2.0) y SDXL
   (OpenRAIL++-M) **si** permiten uso comercial, pero son insuficientes por si
   solas porque el framework, los pesos base y los datos son NC.
4. **OpenRAIL++-M no es totalmente libre:** permite uso comercial pero impone
   restricciones de uso (no generar contenido ilegal, danino, desinformacion,
   etc.). Hay que aceptar y respetar esas condiciones.
5. **SCHP es MIT en codigo**, pero los *checkpoints* se entrenaron sobre LIP /
   ATR / Pascal-Person-Part, datasets con sus propias condiciones. Para uso
   comercial hay que **reentrenar o verificar** la licencia de esos pesos.

## Que significa "no comercial" en la practica

Prohibido, sin licencia adicional:

- Vender el modelo o el software que lo incorpora.
- Ofrecer try-on como SaaS o API de pago.
- Usarlo dentro de un producto/servicio de una empresa (aunque sea gratuito).
- Usarlo en publicidad, marketing o generacion de catalogos comerciales.

Permitido normalmente (I+D no comercial):

- Investigacion, evaluacion interna, prototipos de laboratorio, docencia.
- Experimentos tecnicos para validar la receta (lo que hace la "Pista A").

## Efecto en cascada sobre el codigo propio del proyecto

Todo lo que vive dentro del repositorio derivado (`train_xl.py` modificado,
`watchdog_entrenamiento.sh`, `gradio_demo/app.py`, scripts de
`configuracion-v*-entrenamiento/`) es una **obra derivada** del codigo
CC BY-NC-SA 4.0. Esto significa que:

- El codigo modificado sigue siendo **no comercial** (clausula NC).
- Si se distribuye, debe hacerse bajo **la misma licencia** (clausula SA).
- No se puede cerrar el codigo ni venderlo sin licencia del autor original.

Por tanto, para un producto comercial **no sirve** simplemente "no vender los
pesos": tambien hay que salir del framework no comercial o negociar licencia.

## Fuentes verificadas (texto oficial)

| Componente | Fuente | Texto clave |
|---|---|---|
| IDM-VTON | `LICENSE.txt` local (raiz del repo) | "Attribution-NonCommercial-ShareAlike 4.0 International" |
| IDM-VTON (README) | `README.md` local, seccion License | "The codes and checkpoints ... are under the CC BY-NC-SA 4.0 license" |
| VITON-HD | https://github.com/shadow2496/VITON-HD | "All material is made available under Creative Commons BY-NC 4.0. You can use, redistribute, and adapt the material for non-commercial purposes" + "dataset for research purposes only" |
| DensePose | https://raw.githubusercontent.com/facebookresearch/DensePose/main/LICENSE | "Attribution-NonCommercial 4.0 International" (CC BY-NC 4.0) |
| OpenPose | https://github.com/CMU-Perceptual-Computing-Lab/openpose | "OpenPose is freely available for free non-commercial use ... Interested in a commercial license? Check this FlintBox link" |
| SCHP | https://github.com/GoGoDuck912/Self-Correction-Human-Parsing | "MIT license" (codigo) |
| IP-Adapter | https://raw.githubusercontent.com/tencent-ailab/IP-Adapter/main/LICENSE | "Apache License Version 2.0, January 2004" |
| SDXL | stabilityai/stable-diffusion-xl-base-1.0 | CreativeML OpenRAIL++-M (uso comercial permitido con restricciones de uso) |

> Recordatorio: las licencias Creative Commons **NC** prohiben explicitamente
> el uso "primarily intended for or directed toward commercial advantage or
> monetary compensation". Las CC **BY-NC-SA** anaden obligacion de compartir
> igual. La comprobacion de que no haya dependencias adicionales ocultas
> (pesos, ONNX, checkpoints de terceros) debe cerrarse con revision legal.
