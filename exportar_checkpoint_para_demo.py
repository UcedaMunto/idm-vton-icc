#!/usr/bin/env python3
"""Merge a compact training checkpoint (manifest.json + trainable_state.pt) onto
the base full-pipeline checkpoint, producing a standalone folder loadable by
gradio_demo/app.py (and switch_model_version.sh custom). Only the unet/
subfolder is regenerated; the rest are symlinked from the base checkpoint since
they are frozen and unchanged.
"""
import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.unet_hacked_tryon import UNet2DConditionModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", required=True, help="Full diffusers-format checkpoint (e.g. result_train_night/checkpoint-250)")
    parser.add_argument("--compact_checkpoint", required=True, help="Compact checkpoint dir with trainable_state.pt to overlay")
    parser.add_argument("--output_dir", required=True, help="Destination folder, ready for IDMVTON_MODEL_PATH")
    return parser.parse_args()


def overlay_compact_checkpoint(unet, compact_checkpoint_dir):
    weights_path = os.path.join(compact_checkpoint_dir, "trainable_state.pt")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Compact checkpoint not found: {weights_path}")
    compact_state = torch.load(weights_path, map_location="cpu")
    current_named = dict(unet.named_parameters())
    missing = sorted(set(compact_state) - set(current_named))
    if missing:
        raise RuntimeError(f"Compact checkpoint has names not present in this UNet: {missing[:5]}")
    applied = 0
    for name, value in compact_state.items():
        current_named[name].data.copy_(value.to(current_named[name].dtype))
        applied += 1
    del compact_state
    print(f"[export] overlaid {applied} tensors from {weights_path}")


def main():
    args = parse_args()
    base_checkpoint = os.path.abspath(args.base_checkpoint)
    compact_checkpoint = os.path.abspath(args.compact_checkpoint)
    output_dir = os.path.abspath(args.output_dir)

    if os.path.exists(output_dir):
        raise SystemExit(f"ERROR: output_dir already exists, refusing to overwrite: {output_dir}")
    os.makedirs(output_dir)

    print(f"[export] loading base unet from {base_checkpoint}/unet")
    unet = UNet2DConditionModel.from_pretrained(base_checkpoint, subfolder="unet", torch_dtype=torch.float32)
    overlay_compact_checkpoint(unet, compact_checkpoint)

    unet_out = os.path.join(output_dir, "unet")
    print(f"[export] saving merged unet to {unet_out}")
    unet.save_pretrained(unet_out)
    del unet

    # Symlink the unchanged frozen subfolders + model_index.json from the base checkpoint.
    for entry in os.listdir(base_checkpoint):
        if entry == "unet":
            continue
        src = os.path.join(base_checkpoint, entry)
        dst = os.path.join(output_dir, entry)
        os.symlink(src, dst)
        print(f"[export] symlinked {entry} -> {src}")

    print(f"[export] done. Ready checkpoint at: {output_dir}")


if __name__ == "__main__":
    main()
