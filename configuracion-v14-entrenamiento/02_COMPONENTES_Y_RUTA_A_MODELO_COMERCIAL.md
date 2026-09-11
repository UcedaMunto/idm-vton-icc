# 02 - Componentes y ruta a un modelo comercial

Fecha: 2026-09-09. Este documento responde a: **que habria que cambiar para
poder comercializar**, sin prometer que sea barato. Se separan dos pistas.

## Pista A - I+D no comercial (se puede hacer YA, con el dataset actual)

- Objetivo: **de-riesgar la receta** (que entrenar, con que augmentacion y LR,
  con que criterio de aceptacion) usando el dataset y la base actuales.
- Legal: si, mientras el resultado se use solo para investigacion interna y no
  se explote comercialmente.
- Ventaja: reutiliza todo lo construido (watchdog, checkpoints, export, A/B).
- Salida: una **receta validada** y un pipeline reproducible que luego se
  aplica a datos/stack comerciales en la Pista B.
- Todo lo producido en la Pista A (pesos incluidos) queda **marcado como NC** y
  no debe pasar a produccion comercial. Conveniente etiquetar las carpetas.

## Pista B - Ruta a un modelo comercial

Hay tres estrategias, no excluyentes:

### Opcion B1 - Licenciar lo que ya existe
Negociar licencia comercial con cada titular: autores de IDM-VTON (CC BY-NC-SA),
KAIST/VITON-HD (CC BY-NC), Meta/DensePose (CC BY-NC) y CMU/OpenPose.
- Pros: conserva todo el trabajo y la calidad ya conocida.
- Contras: 4 negociaciones independientes, coste/tiempo inciertos, no siempre
  hay canal. Poco realista como plan principal.
- Estado: **no iniciado** (requiere contacto comercial/legal).

### Opcion B2 - Build limpio (RECOMENDADA)
Reconstruir el stack con **solo componentes permisivos** y datos propios:

| Pieza actual (NC) | Sustituto permisivo | Esfuerzo |
|---|---|---|
| Codigo IDM-VTON (`src/`, `train_xl.py`, `gradio_demo/`) | Reimplementacion propia (clean-room) sobre diffusers + paper | Alto |
| Pesos `yisol/IDM-VTON` | `stabilityai/stable-diffusion-xl-base-1.0` + inpainting (OpenRAIL++-M) | Medio |
| Dataset VITON-HD | Datos propios o licenciados (ver doc 03) | Alto (operativo) |
| DensePose (input) | Pose/representacion propia con licencia permisiva | Medio |
| OpenPose (pose) | Alternativa Apache-2.0 (p.ej. MediaPipe) o modelo propio | Bajo-Medio |
| SCHP (parsing) | Reentrenar/verificar pesos o parser propio | Medio |
| IP-Adapter | Igual (Apache-2.0) | Nulo |
| SDXL | Igual (OpenRAIL++-M) | Nulo |

- Pros: resultado **100% comercializable**, control total del stack.
- Contras: requiere ingenieria (reimplementacion) + recoleccion de datos.
- Clave tecnica: la calidad de IDM-VTON viene de su GarmentNet + IP-Adapter +
  entrenamiento a 768x1024. Un build limpio debe reproducir esas piezas.

### Opcion B3 - Usar un servicio/modelo de terceros YA licenciado
Integrar una API comercial de try-on en lugar de mantener el modelo.
- Pros: cero riesgo de licencia, time-to-market inmediato.
- Contras: dependencia y coste por uso; no se "posee" el modelo.
- Encaja si el negocio es la app/UX y no el modelo en si.

## Arquitectura objetivo (B2) - resumen

```
Inferencia:
  persona -> [parser permisivo] -> mascara agnostica
          -> [pose permitivo]    -> representacion de pose
  prenda  -> [CLIP image encoder Apache-2.0] -> IP-Adapter (Apache-2.0)
  todo    -> SDXL inpainting (OpenRAIL++-M) con GarmentNet entrenado con datos propios

Entrenamiento:
  dataset 100% propio/licenciado (misma estructura que doc 03)
  base: SDXL base/inpainting (comercial)
  entrenar: GarmentNet + IP-Adapter (y/o LoRA) con la receta de doc 04
```

## Decision recomendada

1. Mantener la **Pista A** funcionando (es legal y barato) para validar receta.
2. En paralelo, iniciar **B2**: reimplementacion y decision de datos (B pura o
   compra). B1 solo como atajo si aparece una via de licencia rapida.
3. No invertir en generar pesos "comerciales" hasta que (a) la receta este
   validada en Pista A y (b) el dataset comercial este disponible y auditado.
