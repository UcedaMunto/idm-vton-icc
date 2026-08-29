import os
import random
import argparse
import json
import shutil
import gc
import time
from datetime import datetime, timezone
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline

from ip_adapter.ip_adapter import Resampler
from diffusers.utils.import_utils import is_xformers_available
from typing import Literal, Tuple,List
import torch.utils.data as data
import math
from tqdm.auto import tqdm
from diffusers.training_utils import compute_snr
import torchvision.transforms.functional as TF



class VitonHDDataset(data.Dataset):
    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "test"],
        order: Literal["paired", "unpaired"] = "paired",
        size: Tuple[int, int] = (512, 384),
    ):
        super(VitonHDDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.size = size


        self.norm = transforms.Normalize([0.5], [0.5])
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.transform2D = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.toTensor = transforms.ToTensor()

        with open(
            os.path.join(dataroot_path, phase, "vitonhd_" + phase + "_tagged.json"), "r"
        ) as file1:
            data1 = json.load(file1)

        annotation_list = [
            # "colors",
            # "textures",
            "sleeveLength",
            "neckLine",
            "item",
        ]

        self.annotation_pair = {}
        for k, v in data1.items():
            for elem in v:
                annotation_str = ""
                for template in annotation_list:
                    for tag in elem["tag_info"]:
                        if (
                            tag["tag_name"] == template
                            and tag["tag_category"] is not None
                        ):
                            annotation_str += tag["tag_category"]
                            annotation_str += " "
                self.annotation_pair[elem["file_name"]] = annotation_str


        self.order = order

        self.toTensor = transforms.ToTensor()

        im_names = []
        c_names = []
        dataroot_names = []


        if phase == "train":
            filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")
        else:
            filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")

        with open(filename, "r") as f:
            for line in f.readlines():
                if phase == "train":
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    if order == "paired":
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot_path)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.flip_transform = transforms.RandomHorizontalFlip(p=1)
        self.clip_processor = CLIPImageProcessor()
    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        # subject_txt = self.txt_preprocess['train']("shirt")
        if c_name in self.annotation_pair:
            cloth_annotation = self.annotation_pair[c_name]
        else:
            cloth_annotation = "shirts"
        
        cloth = Image.open(os.path.join(self.dataroot, self.phase, "cloth", c_name))

        im_pil_big = Image.open(
            os.path.join(self.dataroot, self.phase, "image", im_name)
        ).resize((self.width,self.height))

        image = self.transform(im_pil_big)
        # load parsing image


        mask = Image.open(os.path.join(self.dataroot, self.phase, "agnostic-mask", im_name.replace('.jpg','_mask.png'))).resize((self.width,self.height))
        mask = self.toTensor(mask)
        mask = mask[:1]
        densepose_name = im_name
        densepose_map = Image.open(
            os.path.join(self.dataroot, self.phase, "image-densepose", densepose_name)
        )
        pose_img = self.toTensor(densepose_map)  # [-1,1]
 


        if self.phase == "train":
            if random.random() > 0.5:
                cloth = self.flip_transform(cloth)
                mask = self.flip_transform(mask)
                image = self.flip_transform(image)
                pose_img = self.flip_transform(pose_img)



            if random.random()>0.5:
                color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.5, hue=0.5)
                fn_idx, b, c, s, h = transforms.ColorJitter.get_params(color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,color_jitter.hue)
                
                image = TF.adjust_contrast(image, c)
                image = TF.adjust_brightness(image, b)
                image = TF.adjust_hue(image, h)
                image = TF.adjust_saturation(image, s)

                cloth = TF.adjust_contrast(cloth, c)
                cloth = TF.adjust_brightness(cloth, b)
                cloth = TF.adjust_hue(cloth, h)
                cloth = TF.adjust_saturation(cloth, s)

              
            if random.random() > 0.5:
                scale_val = random.uniform(0.8, 1.2)
                image = transforms.functional.affine(
                    image, angle=0, translate=[0, 0], scale=scale_val, shear=0
                )
                mask = transforms.functional.affine(
                    mask, angle=0, translate=[0, 0], scale=scale_val, shear=0
                )
                pose_img = transforms.functional.affine(
                    pose_img, angle=0, translate=[0, 0], scale=scale_val, shear=0
                )



            if random.random() > 0.5:
                shift_valx = random.uniform(-0.2, 0.2)
                shift_valy = random.uniform(-0.2, 0.2)
                image = transforms.functional.affine(
                    image,
                    angle=0,
                    translate=[shift_valx * image.shape[-1], shift_valy * image.shape[-2]],
                    scale=1,
                    shear=0,
                )
                mask = transforms.functional.affine(
                    mask,
                    angle=0,
                    translate=[shift_valx * mask.shape[-1], shift_valy * mask.shape[-2]],
                    scale=1,
                    shear=0,
                )
                pose_img = transforms.functional.affine(
                    pose_img,
                    angle=0,
                    translate=[
                        shift_valx * pose_img.shape[-1],
                        shift_valy * pose_img.shape[-2],
                    ],
                    scale=1,
                    shear=0,
                )




        mask = 1-mask

        cloth_trim =  self.clip_processor(images=cloth, return_tensors="pt").pixel_values


        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        im_mask = image * mask

        pose_img =  self.norm(pose_img)


        result = {}
        result["c_name"] = c_name
        result["image"] = image
        result["cloth"] = cloth_trim
        result["cloth_pure"] = self.transform(cloth)
        result["inpaint_mask"] = 1-mask
        result["im_mask"] = im_mask
        result["caption"] = "model is wearing " + cloth_annotation
        result["caption_cloth"] = "a photo of " + cloth_annotation
        result["annotation"] = cloth_annotation
        result["pose_img"] = pose_img


        return result

    def __len__(self):
        return len(self.im_names)


class ResumableShuffleSampler(data.Sampler):
    """V11 fix: a fixed, seeded shuffle order over the whole dataset, rotated to
    start at `start_index`. This lets a chain of independent process runs (each
    resuming only weights/optimizer, never the dataloader position) continue
    from where the previous run left off instead of silently restarting at
    sample 0 every time (see configuracion-v11-entrenamiento/01_CAUSA_RAIZ_REPETICION_DATOS.md).
    """

    def __init__(self, data_source, start_index=0, seed=0):
        self.num_samples = len(data_source)
        self.start_index = start_index % self.num_samples if self.num_samples else 0
        self.seed = seed

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        order = torch.randperm(self.num_samples, generator=generator).tolist()
        rotated = order[self.start_index:] + order[: self.start_index]
        return iter(rotated)

    def __len__(self):
        return self.num_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",required=False,help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--pretrained_garmentnet_path",type=str,default="stabilityai/stable-diffusion-xl-base-1.0",required=False,help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--checkpointing_epoch",type=int,default=10,help=("Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"" training using `--resume_from_checkpoint`."),)
    parser.add_argument("--checkpointing_steps", type=int, default=None, help="Save a checkpoint every X completed optimizer updates. Defaults to checkpointing_epoch for backwards compatibility.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume trainable weights and optimizer state from a compact V3 checkpoint.")
    parser.add_argument("--resume_optimizer_state", action="store_true", help="Also load optimizer state when resuming; this costs several GiB of RAM.")
    parser.add_argument("--garmentnet_dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"], help="CPU dtype for the frozen GarmentNet encoder.")
    parser.add_argument("--full_pipeline_checkpoint", action="store_true", help="Also save a complete inference pipeline at checkpoint intervals. This can require more than 30 GiB RAM.")
    parser.add_argument("--pretrained_ip_adapter_path",type=str,default="ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin",help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",)
    parser.add_argument("--image_encoder_path",type=str,default="ckpt/image_encoder",required=False,help="Path to CLIP image encoder",)
    parser.add_argument("--gradient_checkpointing",action="store_true",help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",)
    parser.add_argument("--width",type=int,default=768,)
    parser.add_argument("--height",type=int,default=1024,)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--logging_steps",type=int,default=1000,help=("Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"" training using `--resume_from_checkpoint`."),)
    parser.add_argument("--output_dir",type=str,default="output",help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--snr_gamma",type=float,default=None,help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. ""More details here: https://arxiv.org/abs/2303.09556.",)
    parser.add_argument("--num_tokens",type=int,default=16,help=("IP adapter token nums"),)
    parser.add_argument("--learning_rate",type=float,default=1e-5,help="Learning rate to use.",)
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--train_batch_size", type=int, default=6, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--test_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=130)
    parser.add_argument("--max_train_steps",type=int,default=None,help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",)
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--mixed_precision",type=str,default=None,choices=["no", "fp16", "bf16"],help=("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="" 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"" flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."),)
    parser.add_argument("--guidance_scale",type=float,default=2.0,)
    parser.add_argument("--seed", type=int, default=42,)    
    parser.add_argument("--num_inference_steps",type=int,default=30,)    
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.0,
        help="Gradient clipping norm. Default 0 disables clipping to avoid fp16/grad-scaler incompatibilities in this stack.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--data_dir", type=str, default="/home/omnious/workspace/yisol/Dataset/VITON-HD/zalando", help="For distributed training: local_rank")
    parser.add_argument("--train_num_workers", type=int, default=8, help="Number of dataloader workers for training.")
    parser.add_argument("--test_num_workers", type=int, default=2, help="Number of dataloader workers for testing.")
    parser.add_argument("--low_vram_training", action="store_true", help="Keep frozen modules on CPU and move only the tensors needed for the trainable UNet to GPU.")
    parser.add_argument(
        "--train_ip_adapter_only",
        action="store_true",
        help="Train only IP-Adapter layers (attention processors + image proj + conv_in) to reduce VRAM during backward.",
    )
    parser.add_argument(
        "--hybrid_small_models_gpu",
        action="store_true",
        help=(
            "V7 Tier 1 (hybrid GPU/CPU): only effective together with --low_vram_training. "
            "Moves vae, text_encoder, text_encoder_2 and image_encoder to GPU in fp16 to speed up their "
            "forward pass. GarmentNet (unet_encoder) and the trainable UNet remain on CPU, unchanged. "
            "Every tensor produced by these models is converted back to CPU float32 immediately after use, "
            "so the rest of the pipeline is identical to --low_vram_training alone."
        ),
    )
    parser.add_argument(
        "--profile_step_timing",
        action="store_true",
        help=(
            "V10 diagnostic only: prints wall-clock timing breakdown (per training-step stage) with a "
            "[profile] prefix. Pure instrumentation, does not change any tensor computation or output."
        ),
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=0,
        help=(
            "V10 fix: overrides torch.set_num_threads() for CPU-bound modules (GarmentNet, trainable UNet, "
            "and VAE/text/image encoders when running on CPU under --low_vram_training). `accelerate launch` "
            "sets OMP_NUM_THREADS=1 by default even for a single-process run, which silently limits PyTorch's "
            "CPU intra-op parallelism to 1 thread regardless of available cores (found via --profile_step_timing). "
            "0 (default) leaves PyTorch's own thread count untouched, i.e. no behavior change unless set explicitly."
        ),
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args





def main():


    args = parse_args()
    if args.checkpointing_steps is None:
        args.checkpointing_steps = args.checkpointing_epoch
    if args.checkpointing_steps <= 0:
        raise ValueError("checkpointing_steps must be greater than zero")
    effective_mixed_precision = args.mixed_precision
    if args.low_vram_training and effective_mixed_precision == "fp16":
        effective_mixed_precision = "no"

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        mixed_precision=effective_mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=accelerator_project_config,
    )

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    if args.profile_step_timing and accelerator.is_main_process:
        accelerator.print(
            f"[profile] torch.get_num_threads()={torch.get_num_threads()} "
            f"os.cpu_count()={os.cpu_count()} OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
            f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}"
        )

    if accelerator.is_main_process:
        if args.low_vram_training and args.mixed_precision == "fp16":
            accelerator.print(
                "[train] disabling mixed precision for low-vram training to avoid AMP scaler incompatibilities on this setup"
            )
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if effective_mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif effective_mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    if args.low_vram_training:
        weight_dtype = torch.float32
        if accelerator.is_main_process:
            accelerator.print("[train] using float32 weights in low-vram mode to keep CPU-side modules compatible")
    cpu_dtype = torch.float32
    garmentnet_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.garmentnet_dtype]

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",rescale_betas_zero_snr=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=cpu_dtype)
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=cpu_dtype)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,subfolder="vae",torch_dtype=cpu_dtype,)
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(args.pretrained_garmentnet_path, subfolder="unet", torch_dtype=garmentnet_dtype)
    unet_encoder.config.addition_embed_type = None
    unet_encoder.config["addition_embed_type"] = None
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path, torch_dtype=cpu_dtype)
    frozen_device = torch.device("cpu") if args.low_vram_training else accelerator.device

    #customize unet start
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",low_cpu_mem_usage=True, device_map=None, torch_dtype=weight_dtype)
    unet.config.encoder_hid_dim = image_encoder.config.hidden_size
    unet.config.encoder_hid_dim_type = "ip_image_proj"
    unet.config["encoder_hid_dim"] = image_encoder.config.hidden_size
    unet.config["encoder_hid_dim_type"] = "ip_image_proj"


    state_dict = torch.load(args.pretrained_ip_adapter_path, map_location="cpu")
 
 
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    adapter_modules.load_state_dict(state_dict["ip_adapter"],strict=True)

    #ip-adapter
    image_proj_model_device = torch.device("cpu") if args.low_vram_training else accelerator.device
    image_proj_model = Resampler(
        dim=image_encoder.config.hidden_size,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=args.num_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
    ).to(image_proj_model_device, dtype=weight_dtype)

    image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
    image_proj_model.requires_grad_(True)

    del state_dict
    gc.collect()

    unet.encoder_hid_proj = image_proj_model

    conv_new = torch.nn.Conv2d(
        in_channels=4+4+1+4,
        out_channels=unet.conv_in.out_channels,
        kernel_size=3,
        padding=1,
    ).to(dtype=weight_dtype)
    torch.nn.init.kaiming_normal_(conv_new.weight)  
    conv_new.weight.data = conv_new.weight.data * 0.  

    old_in_channels = unet.conv_in.weight.data.shape[1]
    if old_in_channels > conv_new.in_channels:
        raise RuntimeError(
            f"pretrained UNet has {old_in_channels} input channels, but target conv expects {conv_new.in_channels}"
        )
    conv_new.weight.data[:, :old_in_channels] = unet.conv_in.weight.data  
    conv_new.bias.data = unet.conv_in.bias.data  

    unet.conv_in = conv_new  # replace conv layer in unet
    unet.config['in_channels'] = 13  # update config
    unet.config.in_channels = 13  # update config
    #customize unet end


    if not args.low_vram_training:
        vae.to(accelerator.device)
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        text_encoder_2.to(accelerator.device, dtype=weight_dtype)
        image_encoder.to(accelerator.device, dtype=weight_dtype)
        unet_encoder.to(accelerator.device, dtype=weight_dtype)
    elif args.hybrid_small_models_gpu:
        accelerator.print(
            "[train][hybrid-tier1] moving vae/text_encoder/text_encoder_2/image_encoder to GPU in fp16; "
            "GarmentNet and the trainable UNet remain on CPU"
        )
        vae.to(accelerator.device, dtype=torch.float16)
        text_encoder.to(accelerator.device, dtype=torch.float16)
        text_encoder_2.to(accelerator.device, dtype=torch.float16)
        image_encoder.to(accelerator.device, dtype=torch.float16)


    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet_encoder.requires_grad_(False)

    if args.train_ip_adapter_only:
        unet.requires_grad_(False)
        adapter_modules.requires_grad_(True)
        unet.encoder_hid_proj.requires_grad_(True)
        unet.conv_in.requires_grad_(True)
    else:
        unet.requires_grad_(True)




    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if args.enable_xformers_memory_efficient_attention and not args.low_vram_training:
        if is_xformers_available():
            import xformers

            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    elif args.enable_xformers_memory_efficient_attention and args.low_vram_training:
        accelerator.print("[train] skipping xformers attention in low-vram mode to avoid cuDNN initialization issues")
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        unet_encoder.enable_gradient_checkpointing()
    unet.train()

    if args.use_8bit_adam and not args.low_vram_training:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        if args.use_8bit_adam and args.low_vram_training:
            accelerator.print(
                "[train] disabling 8-bit Adam in low-vram mode because bitsandbytes requires CUDA parameters"
            )
        optimizer_class = torch.optim.AdamW

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters were found. Check training flags.")


    optimizer = optimizer_class(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    resume_cumulative_steps = 0
    if args.resume_from_checkpoint:
        weights_path = os.path.join(args.resume_from_checkpoint, "trainable_state.pt")
        optimizer_path = os.path.join(args.resume_from_checkpoint, "optimizer_state.pt")
        manifest_path = os.path.join(args.resume_from_checkpoint, "manifest.json")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Compact checkpoint weights not found: {weights_path}")

        compact_state = torch.load(weights_path, map_location="cpu")
        current_trainable = {name: parameter for name, parameter in unet.named_parameters() if parameter.requires_grad}
        missing_names = sorted(set(current_trainable) - set(compact_state))
        unexpected_names = sorted(set(compact_state) - set(current_trainable))
        if missing_names or unexpected_names:
            raise RuntimeError(
                f"Compact checkpoint parameters do not match. Missing={missing_names[:3]}, unexpected={unexpected_names[:3]}"
            )
        for name, value in compact_state.items():
            current_trainable[name].data.copy_(value)
        del compact_state
        gc.collect()
        if args.resume_optimizer_state and os.path.isfile(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        accelerator.print(f"[train] resumed trainable weights: {args.resume_from_checkpoint}")
        if args.resume_optimizer_state:
            accelerator.print("[train] optimizer state resume enabled")

        # V11 fix: carry the dataset position forward across chained runs (see
        # configuracion-v11-entrenamiento). Old checkpoints without this field
        # fall back to 0 (dataset restarts at the beginning, same as before).
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                resumed_manifest = json.load(manifest_file)
            resume_cumulative_steps = int(resumed_manifest.get("cumulative_steps", 0))
            accelerator.print(f"[train] resumed dataset position: cumulative_steps={resume_cumulative_steps}")
        else:
            accelerator.print(
                "[train] WARNING: resumed checkpoint has no manifest.json; dataset position starts at 0 "
                "(this checkpoint predates the V11 fix)."
            )

    total_params = sum(p.numel() for p in unet.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    accelerator.print(
        f"[train] trainable params: {trainable_count}/{total_params} ({100.0 * trainable_count / total_params:.2f}%)"
    )
    
    train_dataset = VitonHDDataset(
        dataroot_path=args.data_dir,
        phase="train",
        order="paired",
        size=(args.height, args.width),
    )
    # V11 fix: previously shuffle=False with no persisted position meant every
    # resumed run restarted at sample 0, so chained short blocks kept re-training
    # on the same first N pairs instead of progressing through the dataset.
    train_sampler = ResumableShuffleSampler(train_dataset, start_index=resume_cumulative_steps, seed=args.seed)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        # pinned memory cannot be swapped; avoid it in low-vram mode to reduce OOM risk
        pin_memory=not args.low_vram_training,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=args.train_num_workers,
    )
    test_dataset = VitonHDDataset(
        dataroot_path=args.data_dir,
        phase="test",
        order="paired",
        size=(args.height, args.width),
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=args.test_num_workers,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    if args.low_vram_training:
        accelerator.print("[train] keeping the heavy stack on CPU during prepare to reduce initial VRAM spikes")
        unet = unet.to(device=torch.device("cpu"))
        image_proj_model = image_proj_model.to(device=torch.device("cpu"))
        optimizer = optimizer
        train_dataloader = train_dataloader
        test_dataloader = test_dataloader
        unet, image_proj_model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
            unet,
            image_proj_model,
            optimizer,
            train_dataloader,
            test_dataloader,
            device_placement=[False, False, False, False, False],
        )
    else:
        unet,image_proj_model,unet_encoder,image_encoder,optimizer,train_dataloader,test_dataloader = accelerator.prepare(unet, image_proj_model,unet_encoder,image_encoder,optimizer,train_dataloader,test_dataloader)
    initial_global_step = 0

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    global_step = 0
    first_epoch = 0
    train_loss=0.0

    def save_checkpoint(step):
        if not accelerator.is_main_process:
            return

        save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
        if os.path.exists(save_path):
            accelerator.print(f"[train] checkpoint already exists, skipping: {save_path}")
            return

        temporary_path = f"{save_path}.tmp"
        if os.path.exists(temporary_path):
            shutil.rmtree(temporary_path)
        os.makedirs(temporary_path)

        # release any unreferenced tensors before the transient pipeline object is built
        gc.collect()
        torch.cuda.empty_cache()

        unwrapped_unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
        trainable_state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in unwrapped_unet.named_parameters()
            if parameter.requires_grad
        }
        optimizer_state = optimizer.state_dict()
        torch.save(trainable_state, os.path.join(temporary_path, "trainable_state.pt"))
        torch.save(optimizer_state, os.path.join(temporary_path, "optimizer_state.pt"))

        if args.full_pipeline_checkpoint:
            pipeline = TryonPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=unwrapped_unet,
                vae=vae,
                scheduler=noise_scheduler,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                image_encoder=image_encoder,
                unet_encoder=unet_encoder,
                torch_dtype=torch.float16,
                add_watermarker=False,
                safety_checker=None,
            )
            pipeline.save_pretrained(temporary_path)
            del pipeline

        del trainable_state, optimizer_state
        gc.collect()
        torch.cuda.empty_cache()

        if not os.path.isfile(os.path.join(temporary_path, "trainable_state.pt")):
            raise RuntimeError(f"Incomplete compact checkpoint: {temporary_path}")

        manifest = {
            "base_checkpoint": args.pretrained_model_name_or_path,
            "completed_optimizer_updates": step,
            "cumulative_steps": resume_cumulative_steps + step,
            "checkpointing_steps": args.checkpointing_steps,
            "checkpoint_type": "full_pipeline" if args.full_pipeline_checkpoint else "compact_training_state",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "arguments": vars(args),
        }
        with open(os.path.join(temporary_path, "manifest.json"), "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2, sort_keys=True)

        os.replace(temporary_path, save_path)
        accelerator.print(f"[train] checkpoint saved: {save_path}")

    # V10 diagnostic only: no-op unless --profile_step_timing is passed.
    _prof_state = {"t": None}
    def _prof_mark(label):
        if not args.profile_step_timing:
            return
        now = time.time()
        if _prof_state["t"] is not None:
            accelerator.print(f"[profile] {label}: {now - _prof_state['t']:.2f}s")
        _prof_state["t"] = now

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet), accelerator.accumulate(image_proj_model):
                if (not args.low_vram_training) and global_step % args.logging_steps == 0:
                    if accelerator.is_main_process:
                        with torch.no_grad():
                            with torch.cuda.amp.autocast():
                                unwrapped_unet= accelerator.unwrap_model(unet)
                                newpipe = TryonPipeline.from_pretrained(
                                    args.pretrained_model_name_or_path,
                                    unet=unwrapped_unet,
                                    vae= vae,
                                    scheduler=noise_scheduler,
                                    tokenizer=tokenizer,
                                    tokenizer_2=tokenizer_2,
                                    text_encoder=text_encoder,
                                    text_encoder_2=text_encoder_2,
                                    image_encoder=image_encoder,
                                    unet_encoder = unet_encoder,
                                    torch_dtype=torch.float16,
                                    add_watermarker=False,
                                    safety_checker=None,
                                ).to(accelerator.device)
                                with torch.no_grad():
                                    for sample in test_dataloader:
                                        img_emb_list = []
                                        for i in range(sample['cloth'].shape[0]):
                                            img_emb_list.append(sample['cloth'][i])

                                        prompt = sample["caption"]

                                        num_prompts = sample['cloth'].shape[0]                                        
                                        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

                                        if not isinstance(prompt, List):
                                            prompt = [prompt] * num_prompts
                                        if not isinstance(negative_prompt, List):
                                            negative_prompt = [negative_prompt] * num_prompts

                                        image_embeds = torch.cat(img_emb_list,dim=0)


                                        with torch.inference_mode():
                                            (
                                                prompt_embeds,
                                                negative_prompt_embeds,
                                                pooled_prompt_embeds,
                                                negative_pooled_prompt_embeds,
                                            ) = newpipe.encode_prompt(
                                                prompt,
                                                num_images_per_prompt=1,
                                                do_classifier_free_guidance=True,
                                                negative_prompt=negative_prompt,
                                            )
                                        
                                        
                                            prompt = sample["caption_cloth"]
                                            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

                                            if not isinstance(prompt, List):
                                                prompt = [prompt] * num_prompts
                                            if not isinstance(negative_prompt, List):
                                                negative_prompt = [negative_prompt] * num_prompts


                                            with torch.inference_mode():
                                                (
                                                    prompt_embeds_c,
                                                    _,
                                                    _,
                                                    _,
                                                ) = newpipe.encode_prompt(
                                                    prompt,
                                                    num_images_per_prompt=1,
                                                    do_classifier_free_guidance=False,
                                                    negative_prompt=negative_prompt,
                                                )
                                            


                                            generator = torch.Generator(newpipe.device).manual_seed(args.seed) if args.seed is not None else None
                                            images = newpipe(
                                                prompt_embeds=prompt_embeds,
                                                negative_prompt_embeds=negative_prompt_embeds,
                                                pooled_prompt_embeds=pooled_prompt_embeds,
                                                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                                                num_inference_steps=args.num_inference_steps,
                                                generator=generator,
                                                strength = 1.0,
                                                pose_img = sample['pose_img'],
                                                text_embeds_cloth=prompt_embeds_c,
                                                cloth = sample["cloth_pure"].to(accelerator.device),
                                                mask_image=sample['inpaint_mask'],
                                                image=(sample['image']+1.0)/2.0, 
                                                height=args.height,
                                                width=args.width,
                                                guidance_scale=args.guidance_scale,
                                                ip_adapter_image = image_embeds,
                                            )[0]

                                        for i in range(len(images)):
                                            images[i].save(os.path.join(args.output_dir,str(global_step)+"_"+str(i)+"_"+"test.jpg"))                                    
                                        break
                        del unwrapped_unet
                        del newpipe                
                        torch.cuda.empty_cache()



                _prof_mark("step_start")
                vae_device = next(vae.parameters()).device
                pixel_values = batch["image"].to(device=vae_device, dtype=vae.dtype)
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor
                if args.low_vram_training:
                    model_input = model_input.to("cpu", dtype=torch.float32)
                _prof_mark("vae_model_input")

                masked_source = batch["im_mask"].reshape(batch["image"].shape).to(device=vae_device, dtype=vae.dtype)

                masked_latents = vae.encode(masked_source).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor
                if args.low_vram_training:
                    masked_latents = masked_latents.to("cpu", dtype=torch.float32)
                _prof_mark("vae_masked_latents")
                masks = batch["inpaint_mask"]
                # resize the mask to latents shape as we concatenate the mask to the latents
                mask = torch.stack(
                    [
                        torch.nn.functional.interpolate(masks, size=(args.height // 8, args.width // 8))
                    ]
                )
                mask = mask.reshape(-1, 1, args.height // 8, args.width // 8)

                pose_source = batch["pose_img"].to(device=vae_device, dtype=vae.dtype)

                pose_map = vae.encode(pose_source).latent_dist.sample()
                pose_map = pose_map * vae.config.scaling_factor
                if args.low_vram_training:
                    pose_map = pose_map.to("cpu", dtype=torch.float32)
                _prof_mark("vae_pose_map")

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)

                bsz = model_input.shape[0]
                timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(model_input, noise, timesteps)

                if pose_map.shape[-2:] != noisy_latents.shape[-2:]:
                    pose_map = torch.nn.functional.interpolate(
                        pose_map,
                        size=noisy_latents.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                if not args.low_vram_training:
                    noisy_latents = noisy_latents.to(accelerator.device)
                    mask = mask.to(accelerator.device)
                    masked_latents = masked_latents.to(accelerator.device)
                    pose_map = pose_map.to(accelerator.device)
                    model_input = model_input.to(accelerator.device)
                    noise = noise.to(accelerator.device)
                    timesteps = timesteps.to(accelerator.device)

                latent_model_input = torch.cat([noisy_latents, mask,masked_latents,pose_map], dim=1)
            
            
                text_input_ids = tokenizer(
                    batch['caption'],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids
                text_input_ids_2 = tokenizer_2(
                    batch['caption'],
                    max_length=tokenizer_2.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids

                if args.low_vram_training:
                    text_encoder_device = next(text_encoder.parameters()).device
                    text_input_ids_device = text_input_ids.to(text_encoder_device)
                else:
                    text_input_ids_device = text_input_ids.to(accelerator.device)

                encoder_output = text_encoder(text_input_ids_device, output_hidden_states=True)
                text_embeds = encoder_output.hidden_states[-2]
                if args.low_vram_training:
                    text_embeds = text_embeds.to("cpu", dtype=torch.float32)
                if args.low_vram_training:
                    text_encoder_2_device = next(text_encoder_2.parameters()).device
                    text_input_ids_2_device = text_input_ids_2.to(text_encoder_2_device)
                else:
                    text_input_ids_2_device = text_input_ids_2.to(accelerator.device)

                encoder_output_2 = text_encoder_2(text_input_ids_2_device, output_hidden_states=True)
                pooled_text_embeds = encoder_output_2[0]
                text_embeds_2 = encoder_output_2.hidden_states[-2]
                if args.low_vram_training:
                    pooled_text_embeds = pooled_text_embeds.to("cpu", dtype=torch.float32)
                    text_embeds_2 = text_embeds_2.to("cpu", dtype=torch.float32)
                encoder_hidden_states = torch.concat([text_embeds, text_embeds_2], dim=-1) # concat
                _prof_mark("text_encoders_main")


                def compute_time_ids(original_size, crops_coords_top_left = (0,0)):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (args.height, args.height) 
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(torch.device("cpu") if args.low_vram_training else accelerator.device)
                    return add_time_ids
                
                add_time_ids = torch.cat(
                    [compute_time_ids((args.height, args.height)) for i in range(bsz)]
                )
                        
                img_emb_list = []
                for i in range(bsz):
                    img_emb_list.append(batch['cloth'][i])
                
                image_embeds = torch.cat(img_emb_list,dim=0)
                if args.low_vram_training:
                    image_encoder_device = next(image_encoder.parameters()).device
                    image_embeds_input = image_embeds.to(image_encoder_device)
                else:
                    image_embeds_input = image_embeds.to(accelerator.device)

                image_embeds = image_encoder(image_embeds_input, output_hidden_states=True).hidden_states[-2]
                if args.low_vram_training:
                    image_embeds = image_embeds.to("cpu", dtype=torch.float32)
                else:
                    image_embeds = image_embeds.to(accelerator.device)
                    image_embeds = image_embeds.to(dtype=weight_dtype)
                ip_tokens = image_proj_model(image_embeds)
                _prof_mark("image_encoder")

                if not args.low_vram_training:
                    model_input = model_input.to(accelerator.device)
                    noise = noise.to(accelerator.device)
                    latent_model_input = latent_model_input.to(accelerator.device)
                    masked_latents = masked_latents.to(accelerator.device)
                    mask = mask.to(accelerator.device)
                    pose_map = pose_map.to(accelerator.device)
                    timesteps = timesteps.to(accelerator.device)
                    encoder_hidden_states = encoder_hidden_states.to(accelerator.device)
                    pooled_text_embeds = pooled_text_embeds.to(accelerator.device)
                    add_time_ids = add_time_ids.to(accelerator.device)
                    ip_tokens = ip_tokens.to(accelerator.device)
            


                # add cond
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}
                unet_added_cond_kwargs["image_embeds"] = ip_tokens

                cloth_values = batch["cloth_pure"].to(device=vae_device, dtype=vae.dtype)
                cloth_values = vae.encode(cloth_values).latent_dist.sample()
                cloth_values = cloth_values * vae.config.scaling_factor
                if args.low_vram_training:
                    cloth_values = cloth_values.to("cpu", dtype=torch.float32)
                _prof_mark("vae_cloth")


                text_input_ids = tokenizer(
                    batch['caption_cloth'],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids
                text_input_ids_2 = tokenizer_2(
                    batch['caption_cloth'],
                    max_length=tokenizer_2.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids

            
                if args.low_vram_training:
                    text_input_ids_cloth_device = text_input_ids.to(text_encoder_device)
                else:
                    text_input_ids_cloth_device = text_input_ids.to(accelerator.device)

                encoder_output = text_encoder(text_input_ids_cloth_device, output_hidden_states=True)
                text_embeds_cloth = encoder_output.hidden_states[-2]
                if args.low_vram_training:
                    text_embeds_cloth = text_embeds_cloth.to("cpu", dtype=torch.float32)
                if args.low_vram_training:
                    text_input_ids_2_cloth_device = text_input_ids_2.to(text_encoder_2_device)
                else:
                    text_input_ids_2_cloth_device = text_input_ids_2.to(accelerator.device)

                encoder_output_2 = text_encoder_2(text_input_ids_2_cloth_device, output_hidden_states=True)
                text_embeds_2_cloth = encoder_output_2.hidden_states[-2]
                if args.low_vram_training:
                    text_embeds_2_cloth = text_embeds_2_cloth.to("cpu", dtype=torch.float32)
                text_embeds_cloth = torch.concat([text_embeds_cloth, text_embeds_2_cloth], dim=-1) # concat
                _prof_mark("text_encoders_cloth")
                with accelerator.autocast():
                    if args.low_vram_training:
                        timesteps_for_unet_encoder = timesteps.cpu()
                    else:
                        timesteps_for_unet_encoder = timesteps

                    garmentnet_inputs = cloth_values.to(dtype=garmentnet_dtype)
                    garmentnet_text = text_embeds_cloth.to(dtype=garmentnet_dtype)
                    down,reference_features = unet_encoder(garmentnet_inputs, timesteps_for_unet_encoder, garmentnet_text, return_dict=False)
                    reference_features = list(reference_features)
                    _prof_mark("garmentnet_forward")

                    if args.low_vram_training:
                        cloth_values = cloth_values.to(accelerator.device)
                        text_embeds_cloth = text_embeds_cloth.to(accelerator.device)
                        reference_features = [feature.to(accelerator.device, dtype=weight_dtype) for feature in reference_features]

                    noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states,added_cond_kwargs=unet_added_cond_kwargs,garment_features=reference_features).sample
                    _prof_mark("unet_forward")


                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                    elif noise_scheduler.config.prediction_type == "sample":
                        # We set the target to latents here, but the model_pred will return the noise sample prediction.
                        target = model_input
                        # We will have to subtract the noise residual from the prediction to get the target sample.
                        model_pred = model_pred - noise
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    
                    if args.snr_gamma is None:
                        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(noise_scheduler, timesteps)
                        if noise_scheduler.config.prediction_type == "v_prediction":
                            # Velocity objective requires that we add one to SNR values before we divide by them.
                            snr = snr + 1
                        mse_loss_weights = (
                            torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                        )

                        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                
                # Backpropagate
                accelerator.backward(loss)
                _prof_mark("backward")

                if accelerator.sync_gradients:
                    if args.max_grad_norm > 0:
                        try:
                            accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                        except ValueError as exc:
                            if "Attempting to unscale FP16 gradients" in str(exc):
                                accelerator.print(
                                    "[train] skipping gradient clipping because fp16 grad-scaler clipping is not supported in this setup"
                                )
                            else:
                                raise

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    _prof_mark("optimizer_step")
                    progress_bar.update(1)
                    global_step += 1
            if accelerator.sync_gradients:
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                if global_step % args.checkpointing_steps == 0:
                    save_checkpoint(global_step)
                accelerator.wait_for_everyone()
            logs = {"step_loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

                
if __name__ == "__main__":
    main()    
