import os
import sys

# Ensure project imports work even when launched outside the repo directory.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PIL import Image
import gradio as gr
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL
from typing import List

import torch
from transformers import AutoTokenizer
import numpy as np
from contextlib import nullcontext
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

FORCE_CPU_MODE = os.environ.get("IDMVTON_FORCE_CPU", "0") == "1"
USE_CUDA = torch.cuda.is_available() and not FORCE_CPU_MODE
device = 'cuda:0' if USE_CUDA else 'cpu'
torch_dtype = torch.float16 if USE_CUDA else torch.float32
LOW_RAM_MODE = os.environ.get("IDMVTON_LOW_RAM", "1") == "1"
SHARE_DEMO = os.environ.get("IDMVTON_SHARE", "0") == "1"
CPU_OFFLOAD = os.environ.get("IDMVTON_CPU_OFFLOAD", "1") == "1"

# 12GB GPUs usually need lower internal resolution to avoid OOM.
TARGET_WIDTH = 576 if LOW_RAM_MODE else 768
TARGET_HEIGHT = 768 if LOW_RAM_MODE else 1024

if USE_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


base_path = 'yisol/IDM-VTON'
example_path = os.path.join(os.path.dirname(__file__), 'example')

unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=LOW_RAM_MODE,
)
unet.requires_grad_(False)
tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=LOW_RAM_MODE,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=LOW_RAM_MODE,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=LOW_RAM_MODE,
    )
vae = AutoencoderKL.from_pretrained(base_path,
                                    subfolder="vae",
                                    torch_dtype=torch_dtype,
                                    low_cpu_mem_usage=LOW_RAM_MODE,
)

# "stabilityai/stable-diffusion-xl-base-1.0",
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=LOW_RAM_MODE,
)

parsing_model = None
openpose_model = None

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
tensor_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )

pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor= CLIPImageProcessor(),
        text_encoder = text_encoder_one,
        text_encoder_2 = text_encoder_two,
        tokenizer = tokenizer_one,
        tokenizer_2 = tokenizer_two,
        scheduler = noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=LOW_RAM_MODE,
)
pipe.unet_encoder = UNet_Encoder
if USE_CUDA:
    # Slice attention heads one at a time — biggest single lever for peak VRAM.
    pipe.enable_attention_slicing(1)
    # Decode the VAE in tiles: drastically cuts decode peak memory.
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    # Decode latents one image at a time (usually only one, but keeps memory flat).
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    # Use xformers memory-efficient attention if available.
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[IDM-VTON] xformers memory-efficient attention enabled.")
    except Exception:
        pass  # xformers not installed — attention_slicing is the fallback
    if CPU_OFFLOAD:
        try:
            pipe.enable_sequential_cpu_offload()
            print("[IDM-VTON] sequential CPU offload enabled.")
        except Exception:
            try:
                pipe.enable_model_cpu_offload()
                print("[IDM-VTON] model CPU offload enabled.")
            except Exception:
                print("[IDM-VTON] CPU offload not available; using standard device placement.")

def start_tryon(dict,garm_img,garment_des,is_checked,is_checked_crop,denoise_steps,seed):
    global parsing_model, openpose_model

    try:
        if is_checked and (parsing_model is None or openpose_model is None):
            parsing_model = Parsing(0)
            openpose_model = OpenPose(0)

        if is_checked:
            openpose_model.preprocessor.body_estimation.model.to(device)
        if not CPU_OFFLOAD:
            pipe.to(device)
            pipe.unet_encoder.to(device)

        tensor_device = device if not CPU_OFFLOAD else "cpu"

        garm_img= garm_img.convert("RGB").resize((TARGET_WIDTH, TARGET_HEIGHT))

        if isinstance(dict, Image.Image):
            human_img_orig = dict.convert("RGB")
            human_layers = None
        else:
            human_img_orig = dict["background"].convert("RGB")
            human_layers = dict.get("layers")
        
        if is_checked_crop:
            width, height = human_img_orig.size
            target_width = int(min(width, height * (3 / 4)))
            target_height = int(min(height, width * (4 / 3)))
            left = (width - target_width) / 2
            top = (height - target_height) / 2
            right = (width + target_width) / 2
            bottom = (height + target_height) / 2
            cropped_img = human_img_orig.crop((left, top, right, bottom))
            crop_size = cropped_img.size
            human_img = cropped_img.resize((TARGET_WIDTH, TARGET_HEIGHT))
        else:
            human_img = human_img_orig.resize((TARGET_WIDTH, TARGET_HEIGHT))


        if is_checked:
            keypoints = openpose_model(human_img.resize((384,512)))
            model_parse, _ = parsing_model(human_img.resize((384,512)))
            mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
            mask = mask.resize((TARGET_WIDTH, TARGET_HEIGHT))
            if USE_CUDA:
                openpose_model.preprocessor.body_estimation.model.to("cpu")
                torch.cuda.empty_cache()
        else:
            if not human_layers:
                raise gr.Error("Manual mask is disabled in low-compatibility mode. Enable auto-mask.")
            mask = pil_to_binary_mask(human_layers[0].convert("RGB").resize((TARGET_WIDTH, TARGET_HEIGHT)))
            # mask = transforms.ToTensor()(mask)
            # mask = mask.unsqueeze(0)
        mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
        mask_gray = to_pil_image((mask_gray+1.0)/2.0)


        human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
         
        

        # Run DensePose on CPU in LOW_RAM_MODE to free GPU for the diffusion pipeline.
        densepose_device = 'cpu' if (LOW_RAM_MODE and USE_CUDA) else ('cuda' if USE_CUDA else 'cpu')
        args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', densepose_device))
        # verbosity = getattr(args, "verbosity", None)
        pose_img = args.func(args,human_img_arg)    
        pose_img = pose_img[:,:,::-1]    
        pose_img = Image.fromarray(pose_img).resize((TARGET_WIDTH, TARGET_HEIGHT))
        
        with torch.no_grad():
            # Extract the images
            with (torch.cuda.amp.autocast() if USE_CUDA else nullcontext()):
                with torch.no_grad():
                    prompt = "model is wearing " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    with torch.inference_mode():
                        (
                            prompt_embeds,
                            negative_prompt_embeds,
                            pooled_prompt_embeds,
                            negative_pooled_prompt_embeds,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=True,
                            negative_prompt=negative_prompt,
                        )
                                        
                        prompt = "a photo of " + garment_des
                        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                        if not isinstance(prompt, List):
                            prompt = [prompt] * 1
                        if not isinstance(negative_prompt, List):
                            negative_prompt = [negative_prompt] * 1
                        with torch.inference_mode():
                            (
                                prompt_embeds_c,
                                _,
                                _,
                                _,
                            ) = pipe.encode_prompt(
                                prompt,
                                num_images_per_prompt=1,
                                do_classifier_free_guidance=False,
                                negative_prompt=negative_prompt,
                            )



                        pose_img =  tensor_transfrom(pose_img).unsqueeze(0).to(tensor_device,torch_dtype)
                        garm_tensor =  tensor_transfrom(garm_img).unsqueeze(0).to(tensor_device,torch_dtype)
                        generator = torch.Generator(tensor_device).manual_seed(seed) if seed is not None else None
                        images = pipe(
                            prompt_embeds=prompt_embeds.to(tensor_device,torch_dtype),
                            negative_prompt_embeds=negative_prompt_embeds.to(tensor_device,torch_dtype),
                            pooled_prompt_embeds=pooled_prompt_embeds.to(tensor_device,torch_dtype),
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(tensor_device,torch_dtype),
                            num_inference_steps=denoise_steps,
                            generator=generator,
                            strength = 1.0,
                            pose_img = pose_img.to(tensor_device,torch_dtype),
                            text_embeds_cloth=prompt_embeds_c.to(tensor_device,torch_dtype),
                            cloth = garm_tensor.to(tensor_device,torch_dtype),
                            mask_image=mask,
                            image=human_img, 
                            height=TARGET_HEIGHT,
                            width=TARGET_WIDTH,
                            ip_adapter_image = garm_img.resize((TARGET_WIDTH, TARGET_HEIGHT)),
                            guidance_scale=2.0,
                        )[0]
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise gr.Error("CUDA sin memoria. Prueba: 1) cerrar apps que usen GPU, 2) mantener Denoising Steps en 20, 3) usar IDMVTON_FORCE_CPU=1 para modo CPU.")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise gr.Error("CUDA sin memoria. Prueba: 1) cerrar apps que usen GPU, 2) mantener Denoising Steps en 20, 3) usar IDMVTON_FORCE_CPU=1 para modo CPU.")
        raise
    finally:
        if USE_CUDA:
            # Guarantee memory is released even when inference fails.
            try:
                if is_checked and openpose_model is not None:
                    openpose_model.preprocessor.body_estimation.model.to("cpu")
            except Exception:
                pass
            # With accelerate CPU offload, modules can be meta tensors;
            # forcing pipe.to("cpu") raises: "Cannot copy out of meta tensor".
            if not CPU_OFFLOAD:
                try:
                    pipe.to("cpu")
                    pipe.unet_encoder.to("cpu")
                except Exception:
                    pass
            torch.cuda.empty_cache()

    if is_checked_crop:
        out_img = images[0].resize(crop_size)        
        human_img_orig.paste(out_img, (int(left), int(top)))    
        return human_img_orig, mask_gray
    else:
        return images[0], mask_gray
    # return images[0], mask_gray

garm_list = os.listdir(os.path.join(example_path,"cloth"))
garm_list_path = [os.path.join(example_path,"cloth",garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path,"human"))
human_list_path = [os.path.join(example_path,"human",human) for human in human_list]

human_ex_list = human_list_path

##default human


image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    gr.Markdown("## IDM-VTON 👕👔👚")
    gr.Markdown("Virtual Try-on with your image and garment image. Check out the [source codes](https://github.com/yisol/IDM-VTON) and the [model](https://huggingface.co/yisol/IDM-VTON)")
    with gr.Row():
        with gr.Column():
            imgs = gr.Image(sources='upload', type="pil", label='Human image (auto-mask mode)')
            with gr.Row():
                is_checked = gr.Checkbox(label="Yes", info="Use auto-generated mask (Takes 5 seconds)",value=True)
            with gr.Row():
                is_checked_crop = gr.Checkbox(label="Yes", info="Use auto-crop & resizing",value=False)

            example = gr.Examples(
                inputs=imgs,
                examples_per_page=10,
                examples=human_ex_list
            )

        with gr.Column():
            garm_img = gr.Image(label="Garment", sources='upload', type="pil")
            with gr.Row(elem_id="prompt-container"):
                with gr.Row():
                    prompt = gr.Textbox(placeholder="Description of garment ex) Short Sleeve Round Neck T-shirts", show_label=False, elem_id="prompt")
            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=8,
                examples=garm_list_path)
        with gr.Column():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
            masked_img = gr.Image(label="Masked image output", elem_id="masked-img",show_share_button=False)
        with gr.Column():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
            image_out = gr.Image(label="Output", elem_id="output-img",show_share_button=False)




    with gr.Column():
        try_button = gr.Button(value="Try-on")
        with gr.Accordion(label="Advanced Settings", open=False):
            with gr.Row():
                denoise_steps = gr.Number(label="Denoising Steps", minimum=12, maximum=40, value=16, step=1)
                seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42)



    try_button.click(fn=start_tryon, inputs=[imgs, garm_img, prompt, is_checked,is_checked_crop, denoise_steps, seed], outputs=[image_out,masked_img], api_name='tryon')

            


image_blocks.launch(server_name='0.0.0.0', server_port=7860, share=SHARE_DEMO, show_api=False, show_error=True)

