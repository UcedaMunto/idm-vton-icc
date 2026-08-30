# 02 - Evaluacion de las causas propuestas por ChatGPT

Fecha: 2026-08-29. Se contrasta el ranking de ChatGPT con la evidencia real de
este proyecto (ver 01). Veredicto por causa.

## 🔴 "Learning rate demasiado alto" -> REBAJADA a MEDIA (riesgo, no causa)

- El sintoma (color desplazado, prenda generica) **ya existia ANTES** de subir el
  LR: aparece con el modelo 1100 (entrenado a LR 1e-5) y con la base. Por lo tanto
  el LR 5e-5 NO lo causa.
- LR 5e-5 (V12) si es un **riesgo** a vigilar (sobreajuste a largo plazo), aunque
  a 1800 pasos el modelo solo se movio 1.39% y el loss no exploto.
- Recomendacion: moderar a 2e-5 como valor seguro, o mantener 5e-5 solo si el
  diagnostico (A/B) muestra que el entrenamiento esta ayudando.

## 🔴 "Cargando incorrectamente el checkpoint entrenado" -> DESCARTADA (con evidencia)

- Export del modelo 1800 verificado: **0 claves faltantes**, IP-adapter (140),
  encoder_hid_proj (51), conv_in 13 canales, y **config IDENTICO** al
  checkpoint_1100 que la app ya usaba bien.
- La app carga `demos/prueba_cumulative_1800` sin error (HTTP 200, 16 GB RSS).
- Si el state_dict no coincide, `from_pretrained` falla/advierte. No hay evidencia
  de eso. -> El checkpoint se carga y usa correctamente.

## 🔴 "Entrenando desde SDXL base/inpainting en vez del IDM-VTON correcto" -> DESCARTADA

- `pretrained_model_name_or_path=result_train_night/checkpoint-250` es un checkpoint
  del proyecto con `in_channels=13` y `encoder_hid_dim_type=ip_image_proj`.
  -> Es arquitectura IDM-VTON, NO SDXL base (que tiene 4 canales).
- Solo el GarmentNet se carga de `stabilityai/stable-diffusion-xl-base-1.0` (eso es
  lo correcto).
- V11 probo que checkpoint-250 se comporta como el oficial yisol/IDM-VTON.
- Matiz valido (no es un bug): podria compararse entrenar desde `yisol/IDM-VTON`
  vs `checkpoint-250` como experimento; pero el punto de partida actual es correcto.

## 🟠 "Normalizacion / VAE / precision incorrecta" -> BAJA (probablemente NO)

- El pipeline upcasta el VAE a float32 (`force_upcast`) en encode/decode; el
  entrenamiento carga VAE en float32. -> Consistente.
- Normalizacion: entrenamiento e inferencia usan `Normalize([0.5],[0.5])`. Igual.
- Escala de latentes: `vae.config.scaling_factor` en ambos. Igual.
- No hay evidencia de mismatch; no es la causa principal.

## 🟠 "Augmentacion de color demasiado agresiva" -> MEDIA-ALTA (CANDIDATO REAL)

- CONFIRMADO: `ColorJitter(brightness=0.5, contrast=0.3, saturation=0.5, hue=0.5)`
  aplicado a imagen Y prenda con p=0.5. `hue=0.5` = +-180 grados.
- Es **identico al proyecto original** (no es un bug nuestro), pero puede ensenar
  al modelo a ser casi invariante al color, lo que colabora al desplazamiento de
  color (azul -> naranja/rojo).
- Recomendacion: reducir la augmentacion (p.ej. hue=0.1, brightness/contrast/
  saturation a ~0.2) como experimento de una variable.

## 🟠 Hipotesis ADDICIONAL (no contemplada por ChatGPT) -> ALTA

**Desajuste de dominio en la entrada.** El modelo se entreno con prendas aisladas
en fondo blanco (formato catalogo VITON-HD). Si la foto real de la prenda del
usuario trae fondo/percha/pliegues, el GarmentNet y el CLIP extraen caracteristicas
incorrectas -> la prenda no se transfiere, y el modelo se apoya en el prompt
generico ("short sleeve round"), de ahi la "camiseta generica + color desplazado".
Dado que el sintoma aparece tambien con el modelo ~base (=oficial segun V11), esta
es la explicacion mas consistente.

## Conclusion del ranking

| Causa (ChatGPT) | Veredicto con evidencia |
|---|---|
| LR alto | MEDIA (riesgo, no causa) |
| Checkpoint mal cargado | DESCARTADA |
| Entrenar desde SDXL base | DESCARTADA |
| Normalizacion/VAE/precision | BAJA |
| Augmentacion de color | MEDIA-ALTA (candidato real) |
| (adicional) desajuste de dominio | ALTA |

El camino de mayor probabilidad de arreglo es **atacar el dominio** (preprocesar la
prenda del usuario aislada/fondo blanco, y/o dataset custom con sus prendas) mas
**reducir la augmentacion de color**. Entrenar mas pasos con la receta actual tiene
techo bajo (ver 01-B).
