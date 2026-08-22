# Version 6: auditoria, paridad y configuracion segura

Fecha de corte: 2026-08-22

## Alcance

Esta carpeta contiene solamente analisis y planes. No activa entrenamientos, no modifica el modelo, no cambia checkpoints y no reemplaza ninguna configuracion existente.

Referencia original comparada:

- `/home/uceda/Documents/IDM-VTON-main`
- Copia sin metadatos Git; se compara por contenido.
- SHA-256 de `train_xl.py`: `84a7e41a76f04f2654a707c3faf79344f169e8e14d010853ca9c06f5b2870e1b`.

Proyecto auditado:

- `/home/uceda/Documents/IDM-VTON`
- Git HEAD: `89aa18d93956f92dc4c995bd8931d330d616c5d5`.
- `train_xl.py` contiene cambios no confirmados al momento de la auditoria.
- SHA-256 auditado de `train_xl.py`: `accdb82f394690b5e9c5857ec8aa8edaf58ee2f75460116b90ff71bb970793c1`.

## Dictamen ejecutivo

No es correcto afirmar que todos los cambios hechos para usar el hardware son neutros para el modelo. Hay correcciones operativas neutras, pero el perfil activo tambien cambia resolucion, precision de GarmentNet, conjunto de parametros entrenables, clipping de gradiente, optimizador efectivo y continuidad del estado del optimizador. Esos cambios pueden alterar los pesos y la calidad final.

Una primera lectura del watchdog sugirio que la automatizacion V5 no coincidia con lo documentado. Una verificacion posterior en vivo (deteniendo una corrida real con SIGTERM) confirmo que el script SI carga `config_v4.env`, SI supervisa la corrida y SI escribe el disyuntor `PAUSAR_POR_ERROR` de forma atomica ante un fallo. El hallazgo original (F-01) se corrigio en la auditoria; el riesgo real remanente es menor: falta validar linaje/hash del checkpoint y migrar la configuracion a un formato no ejecutable para la interfaz.

## Estado de V6

`SOLO_DOCUMENTACION - NO ACTIVAR`

No se incluye `config_v6.env`, launcher ni interfaz ejecutable para impedir una activacion accidental antes de cerrar los fallos bloqueantes.

## Documentos

1. [01_AUDITORIA_CAMBIOS_Y_FALLOS.md](01_AUDITORIA_CAMBIOS_Y_FALLOS.md): cambios contra el original, fallos y severidad.
2. [02_MATRIZ_PARIDAD_Y_PARAMETROS.md](02_MATRIZ_PARIDAD_Y_PARAMETROS.md): separacion entre cambios operativos y cambios matematicos.
3. [03_PLAN_CORRECCIONES_V6.md](03_PLAN_CORRECCIONES_V6.md): orden de correccion, pruebas y retroceso.
4. [04_PLAN_COMPARACION_CALIDAD.md](04_PLAN_COMPARACION_CALIDAD.md): protocolo A/B para demostrar que no se degrada el modelo.
5. [05_ESPECIFICACION_INTERFAZ.md](05_ESPECIFICACION_INTERFAZ.md): pequena interfaz propuesta para editar parametros con validacion y vista previa.
6. [06_TRAZABILIDAD_Y_EVIDENCIAS.md](06_TRAZABILIDAD_Y_EVIDENCIAS.md): inventario, hashes y comandos reproducibles.
7. [07_DECISIONES_Y_EJECUCION.md](07_DECISIONES_Y_EJECUCION.md): registro de las 4 recomendaciones adoptadas y de la detencion ordenada del entrenamiento ejecutada el 2026-08-22.
8. [08_PRIMERAS_PRUEBAS_V6.md](08_PRIMERAS_PRUEBAS_V6.md): primera validacion estatica V6 (sin GPU) y siguiente paso pendiente de confirmacion.

## Artefactos ejecutables (solo validacion, no entrenamiento)

- `config_v6.env`: configuracion V6 revision 1, `MODE=warm_start_weights`.
- `validar_v6.sh`: valida precondiciones e imprime el comando efectivo sin ejecutarlo. No esta instalado en cron.

## Reglas de seguridad

- No usar V6 para entrenar hasta cerrar todos los hallazgos `BLOQUEANTE` y `ALTO`.
- No mezclar checkpoints compactos de lineas distintas solo porque coincidan los nombres de tensores.
- No promover un checkpoint por perdida de entrenamiento solamente; se exige evaluacion visual fija y comparacion contra controles.
- No cambiar simultaneamente dos parametros que afecten la trayectoria matematica.
- No ejecutar Gradio y entrenamiento a la vez en la RTX 3060 de 12 GB.
- No borrar resultados historicos como parte de la migracion.
- Toda interfaz debe generar una vista previa del comando y requerir confirmacion; nunca debe iniciar automaticamente al guardar.

## Resultado esperado

V6 debe convertir el estado actual en un proceso auditable: configuracion unica, reanudacion verificable, automatizacion con disyuntor real, parametros clasificados por impacto y evidencia A/B antes de declarar equivalencia o mejora.
