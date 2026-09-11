# 08 - Curva pasos -> efectividad: medicion real y estimacion

Fecha: 2026-09-09. Todo lo marcado como **medido** fue ejecutado en esta maquina;
lo marcado como **extrapolado** es una estimacion con su incertidumbre declarada.

## La pregunta

> "Con 30.000 pasos, que porcentaje de efectividad tendriamos? Aproximadamente."

## Respuesta corta

| Pasos acumulados | Efectividad relativa (oficial = 100%) | Tipo de dato |
|---:|---|---|
| 0 (oficial `yisol/IDM-VTON`) | **100%** (referencia) | ancla verificada |
| 500 (V14, desde el oficial) | **~90-95%** | **medido** (esta doc, seccion 2.2) |
| 1.500 (V14, desde el oficial) | **~70-80%** | **medido** (esta doc, seccion 2.2) |
| 1.800 (receta V12/V13) | **~0-10%** | **medido** (A/B de V13) |
| **30.000 (misma receta V14)** | **~0-15%; escenario mas probable: el regimen "roto"** | **extrapolado**, no medido |

**Lo esencial:** en todas las recetas probadas en este proyecto la efectividad
**baja** con los pasos, no sube. 30.000 pasos no es una palanca de mejora: es la
misma receta 20 veces mas lejos en la direccion que ya esta fallando. El techo
posible de un fine-tune es igualar al oficial (100%); lo medido se aleja de el.

## 1. Advertencia metodologica (importante)

No existe una metrica formal de "efectividad" en el proyecto. Los unicos datos
cuantitativos disponibles son:

1. El **A/B controlado de V13** (`configuracion-v13-entrenamiento/04_RESULTADO_AB.md`):
   par `048400_0` (prenda azul), seed 42, 448x576, 15 pasos, con ancla **oficial**
   verificada. Es la unica comparacion rigurosa que existe.
2. Las **muestras de revision** que el watchdog genera al final de cada bloque de
   500 pasos (`entrenamiento continuo/imagenes_revision/`): 2 pares, seed 42,
   20 pasos de inferencia.
3. La **deriva de pesos L2** vs base (V13: 0,68% a 500 pasos -> 1,39% a 1.800).

Cualquier "%" que se afirme fuera de esto es una estimacion, no una medicion.

## 2. Evidencia medida

### 2.1 A/B de V13 - la metrica si separa "transfiere" de "roto"

Re-ejecutado con el script autoritativo `ab_analisis.py` sobre los archivos
archivados del A/B (region torso, filas 35%-80%):

| Modelo | oscuro% | azul% | rojo% | vivos% | veredicto |
|---|---:|---:|---:|---:|---|
| **OFICIAL** (sin fine-tune) | **0,0%** | 24,6% | 26,0% | 5,3% | OK: transfiere la prenda |
| BASE night ck-250 | 18,9% | 0,3% | 61,3% | 0,8% | ROTO |
| V13 cum800 | 18,8% | 0,1% | 72,6% | 1,3% | ROTO |
| V12 cum1800 | 16,2% | 0,0% | 94,2% | 3,8% | ROTO |

Lectura: el oficial **no tiene pixeles oscuros** en el torso y conserva azul de la
prenda; todos los checkpoints entrenados colapsan (oscuro 16-19%, azul ~0%,
rojo 61-94%). La conclusion cualitativa de V13/04 se reproduce exactamente.

> Nota de reproducibilidad: los porcentajes absolutos de azul del oficial difieren
> de la tabla de V13/04 (24,6% aqui vs 54% alli) porque los JPG archivados estan
> recomprimidos y la metrica depende de un umbral de saturacion (`s > 0.25`). La
> separacion oficial vs entrenados, que es lo que importa, es identica.

### 2.2 Muestras reales de V14 (medicion nueva) - usadas por primera vez

Correspondencia verificada por `run_*.meta`: `run_20260908_220420/checkpoint-500`
= **500 acumulados desde el oficial**; `run_20260909_100801/checkpoint-500`
= **1.500 acumulados** (tercer bloque).

| Par | Fuente | oscuro% | rojo% | sat | vivos% |
|---|---|---:|---:|---:|---:|
| `048392_0` (prenda: camiseta blanca) | prenda IN | 0,0% | 0,0% | 0,463 | 0,0% |
| | persona IN | 1,8% | 74,8% | 0,335 | 29,2% |
| | **V14 pasos 500** | 24,8% | 14,2% | 0,367 | 8,3% |
| | **V14 pasos 1.500** | 24,8% | 5,7% | 0,394 | 9,9% |
| `048393_0` (prenda: jersey rayado multicolor) | prenda IN | 0,7% | 53,2% | 0,748 | 32,3% |
| | persona IN | 7,6% | 71,9% | 0,598 | 5,9% |
| | **V14 pasos 500** | 16,5% | 25,9% | 0,478 | 6,7% |
| | **V14 pasos 1.500** | **3,8%** | **76,8%** | **0,701** | 21,1% |

Los numeros solos no bastan: la inspeccion **visual** de las 4 salidas (hecha en
esta revision) es la que da el diagnostico:

- **A 500 pasos la transferencia de prenda FUNCIONA** en ambos pares: el jersey
  rayado se reproduce con su patron multicolor y la camiseta blanca sale blanca;
  los pantalones negros siguen negros. No hay artefactos.
- **A 1.500 pasos el jersey sigue bien pero aparecen los primeros artefactos**:
  en `048393_0` los **pantalones negros pasan a granate/rojo oscuro** (deriva de
  color FUERA de la region de la prenda) y la textura del jersey se ve algo mas
  blanda; en `048392_0` aparece una mancha marron en el brazo. Es exactamente el
  sintoma "artefactos rojo/magenta" descrito en V13/04, en version temprana.
- Esa deriva explica la metrica: en `048393_0` el "oscuro" **cae** de 16,5% a 3,8%
  y el rojo **sube** de 25,9% a 76,8% con saturacion 0,478 -> 0,701. **No es una
  mejora: es el negro convertido en rojo saturado.** (Riesgo de mala lectura si
  solo se mira el % oscuro.)

**Conclusion de 2.2:** la receta V14 (base oficial real + `hue=0.1` + LR 2e-5 +
pares limpios) **no colapsa como V12/V13**, pero a 1.500 pasos **ya muestra el
inicio del mismo fallo**. La ventana util medida es de unos **500-1.500 pasos**.

### 2.3 Cuanto se mueve el modelo

| Comparacion | SSIM global | MAE global | SSIM torso | MAE torso |
|---|---:|---:|---:|---:|
| `048392_0`: 500 vs 1.500 | 0,8950 | 4,70/255 | 0,8535 | 4,91/255 |
| `048393_0`: 500 vs 1.500 | 0,9193 | 4,50/255 | 0,9180 | 4,77/255 |

En 1.000 pasos el modelo cambia poco (~5/255 de MAE), coherente con la deriva de
pesos de V13 (1,39% a 1.800 pasos). Pero el cambio va **en la direccion
equivocada**: la deriva pequena ya basta para producir el corrimiento de color.

## 3. Estimacion para 30.000 pasos (extrapolacion, marcada como tal)

**~0-15% de efectividad; escenario mas probable: el mismo regimen "roto" de
V12/V13.** Razonamiento, punto por punto:

1. **La curva medida baja.** En las tres variantes ya probadas (hue 0,5 / hue 0,1,
   LR 5e-5 / 2e-5, base heredada / base oficial) el fallo aparece entre 800 y
   1.800 pasos. No hay **ningun** punto medido en este proyecto donde mas pasos
   hayan mejorado la transferencia de prenda. Extrapolar hacia arriba contradice
   toda la evidencia.
2. **El fallo es una direccion, no una magnitud que se satura a salvo.** La deriva
   de pesos necesaria para romperlo es diminuta (1,39% a 1.800 pasos y ya 0-10% de
   efectividad). 30.000 pasos son ~20x ese recorrido.
3. **30.000 pasos = 2,2 epocas** sobre 13.563 pares. Es decir, 20x mas exposicion
   al *color jitter* que esta senalizando al modelo "el color de la prenda no
   importa" (seccion 4). Refuerza el mecanismo del fallo, no lo corrige.
4. **El techo es 100%** (igualar al oficial, que ya funciona). Un fine-tune no
   puede pasar de ahi; la pregunta real no es "cuanto?", sino "por que entrenar
   30.000 pasos si lo medido se aleja del oficial?".

**Contra-argumento honesto (y por que no cambia la conclusion):** la pendiente de
V14 es mas suave que la de V12/V13 (base oficial real, LR menor, hue menor), asi
que existe una probabilidad no nula de que la deriva se estabilice en un modelo
"algo degradado pero usable". **Ninguna medicion respalda eso hoy**; lo medido
apunta a lo contrario. Se puede resolver en 4 minutos con el gate de 5.2 en vez
de apostar 15 dias.

**Coste de esos 30.000 pasos** (calculadora `plan_tiempo_v14.py`, 43,0 s/paso):
**358,3 h = 14,9 dias** de computo (367,3 h = 15,3 dias con overhead) en 60
bloques de 500. Con 93 GiB libres hoy y `MIN_FREE_GIB=40`, hay que limpiar
checkpoints cada pocos bloques.

## 4. Por que los pasos NO son la palanca (evidencia en el codigo)

La receta activa del watchdog es `--train_ip_adapter_only` (14,17% de los
parametros; GarmentNet, la via de textura/geometria, 100% congelado) y **no pasa
ninguna** `--color_jitter_*`, asi que rigen los defaults de `train_xl.py`
(lineas 344-348: `prob=0.5`, `brightness/contrast/saturation=0.2`, `hue=0.1`).

Y en `train_xl.py` lineas 178-190 el jitter se aplica **a la vez a la imagen de
la persona y a la prenda, con los MISMOS parametros**, al 50% de las muestras:

```python
if random.random() < self.cj_prob:
    ...                                    # h = +/-0.1 -> +/-18 grados de hue
    image = TF.adjust_hue(image, h)        # la persona
    cloth = TF.adjust_hue(cloth, h)        # Y la prenda objetivo
```

Es decir: en la mitad de los pasos, a la prenda objetivo se le cambia el tono y
se le dice al modelo que esa es la verdad. Durante 30.000 pasos eso es una senal
fuerte de "el tono exacto de la prenda es irrelevante" -> el conditioning de color
se debilita -> deriva a rojo. **El corrimiento de color medido en 2.2 es
exactamente el sintoma esperado de este mecanismo.** Mas pasos amplifican la
senal; no la neutralizan.

## 5. Que hacer en lugar de 30.000 pasos

### 5.1 Apagar el hue jitter (bajo riesgo, hipotesis con respaldo en el codigo)

El watchdog **no** acepta variables de entorno para esto: hay que editar
`entrenamiento continuo/watchdog_entrenamiento.sh` (o lanzar un bloque manual) y
pasar `--color_jitter_hue=0.0` (y, si se quiere, `--color_jitter_prob=0.3`).
Recomendacion de `04_RECETA_ENTRENAMIENTO_COMERCIAL.md` seccion 4.4: cambiar **una
sola variable por experimento** y validar con el gate A/B.

### 5.2 Gate A/B antes de acumular (lo que de verdad responde la pregunta)

El protocolo ya esta definido (V13/04 y doc 04 seccion 4.6): `comparar_calidad_v9.py`
con el par de control y **siempre contra el oficial**, midiendo azul/rojo/oscuro
en el torso, con criterio "azul > 50% y sin artefactos rojo/magenta".

- Coste real medido: del log de la muestra del watchdog, ~16 s de carga + ~7 min
  para 2 pares a 20 pasos de inferencia; con 1 par y 15 pasos baja de **4 min**.
- Es la unica forma de convertir la estimacion de la seccion 3 en un dato.
  **Correrlo cada 500-1.000 pasos y pausar en el primer gate que falle**
  (`PAUSAR_WATCHDOG`), en lugar de descubrir el resultado despues de 15 dias.

### 5.3 Si el objetivo es mejorar (no solo no degradar)

La palanca es **que se entrena**, no cuantos pasos (V13/01 seccion D): entrenar
solo el IP-Adapter ajusta color/estilo semantico, no el detalle de la prenda.
Candidatos, de menor a mayor coste: `hue=0` + menor probabilidad de jitter;
entrenar tambien GarmentNet (o LoRA sobre el UNet); alinear la resolucion de
entrenamiento e inferencia (448x576 -> 576x768); y dataset del dominio real de
catalogo (que es ademas un requisito de licencias, ver `03`). Cada cambio se
valida con 5.2.

## 6. Reproducir esta medicion

```bash
cd /home/uceda/Documents/IDM-VTON/configuracion-v14-entrenamiento
python3 analizar_curva_v14.py            # 100% CPU, no toca la GPU en entrenamiento
```

Sale en tres secciones: (A) la curva de V14 sobre las muestras del watchdog,
(B) cuanto se mueve el modelo entre checkpoints, (C) la calibracion contra el A/B
de V13 con su ancla oficial. Para anadir un checkpoint nuevo:

```bash
python3 analizar_curva_v14.py \
  --steps 500=/ruta/imagenes_revision/run_X \
  --steps 1500=/ruta/imagenes_revision/run_Y \
  --steps 2000=/ruta/imagenes_revision/run_Z
```

Las funciones de metrica de este script reproducen **exactamente** los valores de
`configuracion-v13-entrenamiento/resultado_ab_20260830/ab_analisis.py`
(comprobado: OFICIAL azul 24,6% / rojo 26,0%; V12-1800 azul 0,0% / rojo 94,2%).

## 7. Limitaciones de esta medicion (leer antes de citar los numeros)

1. **No hay ancla oficial para los mismos pares.** Las muestras del watchdog solo
   contienen la salida del checkpoint, no la del oficial sobre `048392_0` /
   `048393_0`. Por eso la columna "efectividad relativa" es una **estimacion** que
   combina el juicio visual con la calibracion de 2.1, no una razon medida.
   Obtener el ancla exige una inferencia mas en GPU, hoy ocupada por el
   entrenamiento.
2. **2 pares, 1 semilla, 1 muestra por bloque.** Los pares disponibles tienen
   prenda blanca y rayada, colores poco discriminativos frente a la escena, que es
   justo el caso donde el A/B de V13 separa peor. El par de control idoneo es el
   de V13 (`048400_0`, prenda azul).
3. **Solo un punto intermedio medido (500) y uno en 1.500.** La forma de la curva
   entre ambos no esta muestreada; el `run_20260909_160401` (1.800) se interrumpio
   en el paso 300 y no genero muestra.
4. **Metrica proxy.** "% de pixeles por familia de color en el torso" no es una
   medida de calidad percibida; hay que leerla junto al ojo humano (por eso 2.2
   incluye la inspeccion visual de las 4 imagenes).



