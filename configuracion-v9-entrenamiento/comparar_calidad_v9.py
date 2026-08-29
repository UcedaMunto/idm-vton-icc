# coding=utf-8
"""V9: igual que comparar_calidad_v7.py, pero con registro de progreso por
paso de denoising (no solo al final), para poder seguir el avance real en
vivo con `tail -f` en vez de esperar a ciegas.

No modifica ningun checkpoint en disco. Solo lee y genera imagenes nuevas.
"""
from typing import List, Literal, Tuple

import argparse
import json
import os
import resource
import time

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")

import numpy as np
import torch
import torch.utils.data as data
import torchvision
import transformers
import diffusers
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from torchvision import transforms
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="V9: comparison with per-step progress logging.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--compact_checkpoint", type=str, default=None)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--num_inference_steps", type=int, default=15)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--unpaired", action="store_true")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=1)
    return parser.parse_args()


def log(msg):
    print(f"[v9][{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pil_to_tensor(images):
    images = np.array(images).astype(np.float32) / 255.0
    images = torch.from_numpy(images.transpose(2, 0, 1))
    return images


class VitonHDTestDataset(data.Dataset):
    def __init__(self, dataroot_path: str, phase: Literal["train", "test"], order: Literal["paired", "unpaired"] = "paired", size: Tuple[int, int] = (576, 448), limit: int = None):
        super().__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.toTensor = transforms.ToTensor()

        with open(os.path.join(dataroot_path, phase, "vitonhd_" + phase + "_tagged.json"), "r") as file1:
            data1 = json.load(file1)

        annotation_list = ["sleeveLength", "neckLine", "item"]
        self.annotation_pair = {}
        for k, v in data1.items():
            for elem in v:
                annotation_str = ""
                for template in annotation_list:
                    for tag in elem["tag_info"]:
                        if tag["tag_name"] == template and tag["tag_category"] is not None:
                            annotation_str += tag["tag_category"] + " "
                self.annotation_pair[elem["file_name"]] = annotation_str

        self.order = order
        im_names, c_names, dataroot_names = [], [], []
        filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")
        with open(filename, "r") as f:
            for line in f.readlines():
                if order == "paired":
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot_path)

        if limit is not None:
            im_names, c_names, dataroot_names = im_names[:limit], c_names[:limit], dataroot_names[:limit]

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.clip_processor = CLIPImageProcessor()

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        cloth_annotation = self.annotation_pair.get(c_name, "shirts")
        cloth = Image.open(os.path.join(self.dataroot, self.phase, "cloth", c_name))
        im_pil_big = Image.open(os.path.join(self.dataroot, self.phase, "image", im_name)).resize((self.width, self.height))
        image = self.transform(im_pil_big)

        mask = Image.open(os.path.join(self.dataroot, self.phase, "agnostic-mask", im_name.replace('.jpg', '_mask.png'))).resize((self.width, self.height))
        mask = self.toTensor(mask)[:1]
        mask = 1 - mask
        im_mask = image * mask

        pose_img = Image.open(os.path.join(self.dataroot, self.phase, "image-densepose", im_name)).resize((self.width, self.height))
        pose_img = self.transform(pose_img)

        return {
            "c_name": c_name,
            "im_name": im_name,
            "image": image,
            "cloth_pure": self.transform(cloth),
            "cloth": self.clip_processor(images=cloth, return_tensors="pt").pixel_values,
            "inpaint_mask": 1 - mask,
            "im_mask": im_mask,
            "caption_cloth": "a photo of " + cloth_annotation,
            "caption": "model is wearing a " + cloth_annotation,
            "pose_img": pose_img,
        }

    def __len__(self):
        return len(self.im_names)


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
    log(f"overlaid {applied} tensors from {weights_path}")


def main():
    args = parse_args()
    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
    log(f"start num_inference_steps={args.num_inference_steps} limit={args.limit} compact_checkpoint={args.compact_checkpoint}")

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(mixed_precision="fp16", project_config=accelerator_project_config)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float16
    t0 = time.time()
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", torch_dtype=weight_dtype)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder", torch_dtype=weight_dtype)
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet_encoder", torch_dtype=torch.float32)
    text_encoder_one = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=weight_dtype)
    tokenizer_one = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", use_fast=False)
    log(f"models loaded in {time.time() - t0:.1f}s")

    if args.compact_checkpoint:
        overlay_compact_checkpoint(unet, args.compact_checkpoint)

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet_encoder.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.eval()
    unet_encoder.eval()

    device = accelerator.device
    vae.to(device)
    unet.to(device)
    image_encoder.to(device)
    text_encoder_one.to(device)
    text_encoder_two.to(device)
    # unet_encoder (GarmentNet) stays on CPU: see configuracion-v7-entrenamiento/04_IMPLEMENTACION_Y_PRUEBAS.md
    log(f"device placement done in {time.time() - t0:.1f}s total")

    def _move(obj, target_device, target_dtype=None):
        if torch.is_tensor(obj):
            if target_dtype is not None and obj.is_floating_point():
                return obj.to(target_device, dtype=target_dtype)
            return obj.to(target_device)
        if isinstance(obj, tuple):
            return tuple(_move(o, target_device, target_dtype) for o in obj)
        if isinstance(obj, list):
            return [_move(o, target_device, target_dtype) for o in obj]
        return obj

    _garmentnet_caller_device = {}
    _garmentnet_call_count = {"n": 0}
    _garmentnet_t0 = {"t": None}

    def _garmentnet_pre_hook(module, args_, kwargs_):
        _garmentnet_t0["t"] = time.time()
        if args_:
            _garmentnet_caller_device["device"] = args_[0].device
            _garmentnet_caller_device["dtype"] = args_[0].dtype
        cpu_args = tuple(_move(a, "cpu", torch.float32) for a in args_)
        cpu_kwargs = {k: _move(v, "cpu", torch.float32) for k, v in kwargs_.items()}
        return cpu_args, cpu_kwargs

    def _garmentnet_post_hook(module, args_, kwargs_, output):
        _garmentnet_call_count["n"] += 1
        dt = time.time() - _garmentnet_t0["t"]
        if _garmentnet_call_count["n"] % 5 == 0 or _garmentnet_call_count["n"] <= 3:
            log(f"garmentnet call #{_garmentnet_call_count['n']} took {dt:.2f}s (cpu, fp32)")
        target_device = _garmentnet_caller_device.get("device", device)
        target_dtype = _garmentnet_caller_device.get("dtype", torch.float16)
        return _move(output, target_device, target_dtype)

    unet_encoder.register_forward_pre_hook(_garmentnet_pre_hook, with_kwargs=True)
    unet_encoder.register_forward_hook(_garmentnet_post_hook, with_kwargs=True)

    test_dataset = VitonHDTestDataset(
        dataroot_path=args.data_dir, phase="test",
        order="unpaired" if args.unpaired else "paired",
        size=(args.height, args.width), limit=args.limit,
    )
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=1)

    pipe = TryonPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet, vae=vae, feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one, text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one, tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler, image_encoder=image_encoder, unet_encoder=unet_encoder,
        torch_dtype=weight_dtype,
    )

    step_state = {"t0": None, "n": args.num_inference_steps}

    def _on_step_end(pipe_self, step_index, timestep, callback_kwargs):
        if step_state["t0"] is None:
            step_state["t0"] = time.time()
        elapsed = time.time() - step_state["t0"]
        rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        log(f"denoise step {step_index + 1}/{step_state['n']} elapsed={elapsed:.1f}s max_rss={rss_mib:.0f}MiB")
        return callback_kwargs

    with torch.no_grad(), torch.cuda.amp.autocast():
        for sample in test_dataloader:
            img_emb_list = [sample['cloth'][i] for i in range(sample['cloth'].shape[0])]
            num_prompts = sample['cloth'].shape[0]
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

            prompt = sample["caption"]
            prompt = [prompt] * num_prompts if not isinstance(prompt, List) else prompt
            neg = [negative_prompt] * num_prompts
            image_embeds = torch.cat(img_emb_list, dim=0)

            log(f"encoding prompts for {sample['im_name'][0]}")
            with torch.inference_mode():
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                    prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=neg,
                )
                prompt_c = sample["caption_cloth"]
                prompt_c = [prompt_c] * num_prompts if not isinstance(prompt_c, List) else prompt_c
                prompt_embeds_c, _, _, _ = pipe.encode_prompt(
                    prompt_c, num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=neg,
                )

            generator = torch.Generator(pipe.device).manual_seed(args.seed) if args.seed is not None else None
            step_state["t0"] = time.time()
            log(f"starting denoising loop, {args.num_inference_steps} steps")
            images = pipe(
                prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=args.num_inference_steps, generator=generator, strength=1.0,
                pose_img=sample['pose_img'], text_embeds_cloth=prompt_embeds_c,
                cloth=sample["cloth_pure"].to(accelerator.device), mask_image=sample['inpaint_mask'],
                image=(sample['image'] + 1.0) / 2.0, height=args.height, width=args.width,
                guidance_scale=args.guidance_scale, ip_adapter_image=image_embeds,
                callback_on_step_end=_on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
            )[0]

            for i in range(len(images)):
                x_sample = pil_to_tensor(images[i])
                torchvision.utils.save_image(x_sample, os.path.join(args.output_dir, sample['im_name'][i]))
            log(f"saved {sample['im_name'][0]}")

    log("done")


if __name__ == "__main__":
    main()
