#!/usr/bin/env python3
"""Merge a compact training checkpoint (manifest.json + trainable_state.pt) onto
the base full-pipeline checkpoint, producing a standalone folder loadable by
gradio_demo/app.py (and switch_model_version.sh custom). Only the unet/
subfolder is regenerated; the rest are symlinked from the base checkpoint since
they are frozen and unchanged. V14: si el compact checkpoint entreno el GarmentNet
(receta C2, --train_garmentnet) tambien se regenera la subcarpeta unet_encoder.
"""
import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref

# V14: prefijo con el que train_xl.py guarda los pesos entrenables del GarmentNet
# (receta C2, --train_garmentnet) dentro de trainable_state.pt.
GARMENTNET_PREFIX = "unet_encoder."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", required=True, help="Full diffusers-format checkpoint (e.g. result_train_night/checkpoint-250)")
    parser.add_argument("--compact_checkpoint", required=True, help="Compact checkpoint dir with trainable_state.pt to overlay")
    parser.add_argument("--output_dir", required=True, help="Destination folder, ready for IDMVTON_MODEL_PATH")
    return parser.parse_args()


def split_compact_state(compact_state):
    """V14: separa un trainable_state.pt en claves del unet y claves del GarmentNet.

    Los checkpoints de la receta C2 (--train_garmentnet en train_xl.py) guardan los
    pesos entrenables del GarmentNet en el MISMO fichero, con el prefijo
    "unet_encoder.". Sin separarlos el export fallaria (nombres desconocidos para el
    unet) y, peor, el unet_encoder se symlinkearia al base: la app serviria el
    GarmentNet OFICIAL en lugar del entrenado.
    """
    unet_state = {k: v for k, v in compact_state.items() if not k.startswith(GARMENTNET_PREFIX)}
    garmentnet_state = {
        k[len(GARMENTNET_PREFIX):]: v
        for k, v in compact_state.items()
        if k.startswith(GARMENTNET_PREFIX)
    }
    return unet_state, garmentnet_state


def overlay_state(module, state, label, strip_prefix=""):
    """Copia `state` (dict nombre -> tensor) sobre los parametros de `module`."""
    current_named = dict(module.named_parameters())
    to_apply = {}
    for name, value in state.items():
        key = name
        if strip_prefix:
            if not name.startswith(strip_prefix):
                continue
            key = name[len(strip_prefix):]
        to_apply[key] = value
    missing = sorted(set(to_apply) - set(current_named))
    if missing:
        raise RuntimeError(f"Compact checkpoint has names not present in {label}: {missing[:5]}")
    applied = 0
    for name, value in to_apply.items():
        current_named[name].data.copy_(value.to(current_named[name].dtype))
        applied += 1
    print(f"[export] overlaid {applied} tensors into {label}")
    return applied


def main():
    args = parse_args()
    base_checkpoint = os.path.abspath(args.base_checkpoint)
    compact_checkpoint = os.path.abspath(args.compact_checkpoint)
    output_dir = os.path.abspath(args.output_dir)

    if os.path.exists(output_dir):
        raise SystemExit(f"ERROR: output_dir already exists, refusing to overwrite: {output_dir}")
    os.makedirs(output_dir)

    weights_path = os.path.join(compact_checkpoint, "trainable_state.pt")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Compact checkpoint not found: {weights_path}")
    compact_state = torch.load(weights_path, map_location="cpu")
    unet_state, garmentnet_state = split_compact_state(compact_state)
    del compact_state
    print(
        f"[export] compact checkpoint: {len(unet_state)} tensores para el unet, "
        f"{len(garmentnet_state)} para el GarmentNet"
    )

    print(f"[export] loading base unet from {base_checkpoint}/unet")
    unet = UNet2DConditionModel.from_pretrained(base_checkpoint, subfolder="unet", torch_dtype=torch.float32)
    overlay_state(unet, unet_state, "unet")

    unet_out = os.path.join(output_dir, "unet")
    print(f"[export] saving merged unet to {unet_out}")
    unet.save_pretrained(unet_out)
    del unet

    # V14: si el compact checkpoint entreno el GarmentNet hay que fusionarlo tambien.
    # Se guarda en fp16 (los pesos entrenados ya son fp16 y el base tambien): sin
    # perdida y sin duplicar el disco del export.
    merged_garmentnet = bool(garmentnet_state)
    if merged_garmentnet:
        print(f"[export] merging {len(garmentnet_state)} GarmentNet tensors")
        garmentnet = UNet2DConditionModel_ref.from_pretrained(
            base_checkpoint, subfolder="unet_encoder", torch_dtype=torch.float16
        )
        overlay_state(garmentnet, garmentnet_state, "unet_encoder", strip_prefix=GARMENTNET_PREFIX)
        garmentnet.save_pretrained(os.path.join(output_dir, "unet_encoder"))
        del garmentnet

    # Symlink the unchanged frozen subfolders + model_index.json from the base checkpoint.
    for entry in os.listdir(base_checkpoint):
        if entry == "unet" or (entry == "unet_encoder" and merged_garmentnet):
            continue
        src = os.path.join(base_checkpoint, entry)
        dst = os.path.join(output_dir, entry)
        os.symlink(src, dst)
        print(f"[export] symlinked {entry} -> {src}")

    print(f"[export] done. Ready checkpoint at: {output_dir}")


if __name__ == "__main__":
    main()
