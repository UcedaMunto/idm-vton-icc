# 07 - Checklist go/no-go y riesgos

Fecha: 2026-09-09. Marcar cada punto antes de invertir en un modelo comercial.

## 7.1 Checklist legal (go/no-go comercial)

- [ ] Revision por abogado del analisis de licencias (`01`).
- [ ] El codigo que se distribuya **no** deriva de IDM-VTON (clean-room) **o** se
      tiene licencia comercial explicita del autor.
- [ ] Base de difusion `SDXL`/`inpainting` (OpenRAIL++-M) aceptada, incluidas sus
      restricciones de uso.
- [ ] Dataset 100% propio o licenciado para uso comercial, con trazabilidad.
- [ ] Consentimiento/cesion de derechos de imagen por persona (RGPD si aplica).
- [ ] Herramientas de anotacion (pose / parsing) con licencia comercial.
- [ ] Auditoria de dependencias ocultas: pesos, `.pkl`, `.onnx`, `.bin` sin NC.
- [ ] Politica de privacidad y base legal para tratar imagenes de personas.
- [ ] No usar marcas/nombres de terceros (IDM-VTON, VITON-HD, DensePose, OpenPose).

## 7.2 Checklist tecnico

- [ ] Receta validada en Pista A: gates A/B de 100 y 500 pasando (doc 04).
- [ ] Decidido **que** se entrena (IP-Adapter + GarmentNet; no solo IP-Adapter).
- [ ] Resolucion alineada entre entrenamiento e inferencia.
- [ ] Augmentacion de color reducida (`hue` 0.0-0.05).
- [ ] Disco con margen (>= 60 GiB libres) y politica de retencion aplicada.
- [ ] Export a pipeline probado en la app con checkpoints reales.
- [ ] Revision visual contra el oficial en >= 5 pares variados (no solo 1).

## 7.3 Matriz de riesgos

| Riesgo | Prob. | Impacto | Mitigacion |
|---|---|---|---|
| Publicar algo derivado de licencias NC | Media | Muy alto (legal) | Pista B + auditoria legal |
| Receta actual degrada la prenda (demostrado V13) | Alta | Alto | Cambiar que se entrena + gates A/B |
| Disco lleno detiene el entrenamiento | Alta | Medio | Retencion (doc 06) |
| Sobreajuste con la misma receta | Media | Medio | Validar test, parar si gates empeoran |
| App y entrenamiento no conviven | Segura | Bajo | Planificar ventanas; cloud para produccion |
| RAM/swap al limite | Media | Medio | No cargar la app en paralelo |
| Coste/tiempo subestimado | Media | Medio | Plan de 05 + GPU cloud para B3 |

## 7.4 Decisiones abiertas (para negocio/legal)

1. ¿**Licenciar** lo existente (B1), **construir limpio** (B2) o **integrar
   terceros** (B3)? Recomendado: B2, con B1 solo como atajo.
2. ¿Datos **propios** (fotografia + consentimiento) o **licenciar** un dataset?
3. ¿El negocio necesita **poseer** el modelo o basta una API externa?
4. ¿Presupuesto para **GPU cloud** en la fase de entrenamiento comercial?
5. ¿Mercado objetivo y jurisdiccion? (afecta a RGPD, derechos de imagen y datos).

## 7.5 Recomendacion

- **Ahora:** mantener la Pista A (I+D no comercial) para validar la receta; es
  legal y de bajo coste, y no bloquea nada.
- **Antes de invertir en comercial:** cerrar B0 (auditoria legal) y elegir entre
  B1/B2/B3.
- **No** generar pesos que se pretendan comerciales hasta tener stack y datos
  comerciales; etiquetar todo lo de Pista A como **NO COMERCIAL**.
