# Trazabilidad y evidencias

## Anclas de auditoria

| Elemento | Valor |
|---|---|
| Fecha | 2026-08-22 |
| Original | `/home/uceda/Documents/IDM-VTON-main` |
| Original Git | No contiene `.git`; no se puede obtener commit local |
| Proyecto | `/home/uceda/Documents/IDM-VTON` |
| HEAD | `89aa18d93956f92dc4c995bd8931d330d616c5d5` |
| Rama | `main` |
| Remoto observado | `git@github.com:UcedaMunto/idm-vton-icc.git` |
| `train_xl.py` original | `84a7e41a76f04f2654a707c3faf79344f169e8e14d010853ca9c06f5b2870e1b` |
| `train_xl.py` auditado | `accdb82f394690b5e9c5857ec8aa8edaf58ee2f75460116b90ff71bb970793c1` |
| Watchdog auditado | `6fdc4e4c5bee62df959d095f082a10cf6cf9b02f7f87e2657e576fda42b56850` |
| Config V4 | `96e1a91155bfd3f6d50af68823b7eae4e768d2a82c3486c4c59101dd03af1780` |

## Estado Git observado

```text
 M train_xl.py
?? configuracion-v3-entrenamiento/
?? configuracion-v4-entrenamiento/
?? configuracion-v5-entrenamiento/
?? entrenamiento continuo/
```

La carpeta V2 ya estaba rastreada o no aparecia como nueva en ese corte. V3-V5 y automatizacion no estaban documentadas por commits, por lo que Git no ofrece historia interna para ellas.

## Commits relevantes

| Commit | Mensaje | Superficie principal |
|---|---|---|
| `986bafc` | cambios para entrenamiento | `.env`, guias |
| `240b9ba` | cambios para bajar uso de la ram | `train_xl.py`, launcher, Resampler, attention |
| `105ece7` | corecciones para entrenamiento con 12gb de cpu | `train_xl_long.sh` |
| `89aa18d` | cambios para entrenamiento y para cambiar al modelo entrenado | Gradio, inferencia, selector de modelo y ajuste train |

## Inventario de diferencias relevantes contra original

- `train_xl.py`, `train_xl.sh`, nuevo `train_xl_long.sh`.
- `ip_adapter/resampler.py`.
- `src/attentionhacked_tryon.py`.
- `gradio_demo/app.py`.
- `inference.sh`, `switch_model_version.sh`.
- Guardas CPU en `preprocess/humanparsing/run_parsing.py` y `preprocess/openpose/run_openpose.py`.
- Comentarios traducidos en `gradio_demo/detectron2/layers/aspp.py`.
- Guias, configuraciones V2-V5 y automatizacion.

No se incluyeron datasets, checkpoints, resultados, logs, caches o backups en la comparacion de codigo.

## Comandos reproducibles de lectura

Estos comandos no modifican archivos:

```bash
git -C /home/uceda/Documents/IDM-VTON status --short
git -C /home/uceda/Documents/IDM-VTON log --oneline --decorate --all -n 30
git -C /home/uceda/Documents/IDM-VTON diff -- train_xl.py
diff -u /home/uceda/Documents/IDM-VTON-main/train_xl.py /home/uceda/Documents/IDM-VTON/train_xl.py
sha256sum \
  /home/uceda/Documents/IDM-VTON-main/train_xl.py \
  /home/uceda/Documents/IDM-VTON/train_xl.py \
  '/home/uceda/Documents/IDM-VTON/entrenamiento continuo/watchdog_entrenamiento.sh' \
  /home/uceda/Documents/IDM-VTON/configuracion-v4-entrenamiento/config_v4.env
```

## Evidencia que falta obtener antes de implementar

- Hash/identidad verificable del upstream original; la copia `IDM-VTON-main` no tiene Git.
- Comando real y configuracion efectiva de cada checkpoint historico.
- Versiones completas de paquetes/driver para cada linea.
- Manifest/checksums de `checkpoint-250` y compactos V3/V4.
- Prueba de igualdad continua vs reanudada.
- Normas de gradiente para decidir clipping.
- Evaluacion fija C1/C2/C3.
- Pico de RAM durante `optimizer.state_dict()` y `torch.save`.
- Confirmacion experimental de tensores/dispositivos efectivos en low-VRAM.

## Regla para actualizar este documento

Toda evidencia nueva debe indicar fecha, comando o procedimiento, salida resumida y artefactos afectados. Si un hash cambia, se agrega una nueva fila; no se sobrescribe el valor histórico.
