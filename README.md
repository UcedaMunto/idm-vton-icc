
<div align="center">
<h1>IDM-VTON: Improving Diffusion Models for Authentic Virtual Try-on in the Wild</h1>

<a href='https://idm-vton.github.io'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2403.05139'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/spaces/yisol/IDM-VTON'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-yellow'></a>
<a href='https://huggingface.co/yisol/IDM-VTON'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>


</div>

This is the official implementation of the paper ["Improving Diffusion Models for Authentic Virtual Try-on in the Wild"](https://arxiv.org/abs/2403.05139).

Star ⭐ us if you like it!

---


![teaser2](assets/teaser2.png)&nbsp;
![teaser](assets/teaser.png)&nbsp;



## Requirements

```
git clone https://github.com/yisol/IDM-VTON.git
cd IDM-VTON

conda env create -f environment.yaml
conda activate idm
```

### Hardware and system requirements

- Linux (recommended for CUDA support)
- Python 3.10
- NVIDIA GPU with CUDA support for practical inference speed
- NVIDIA driver installed and visible via `nvidia-smi`
- Enough RAM/VRAM for SDXL-based try-on workloads (CPU-only mode is supported but much slower)

### Verify if IDM-VTON is using GPU

Run these checks after activating the `idm` environment:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); print('torch_cuda', torch.version.cuda); print('gpus', torch.cuda.device_count())"
```

Expected for GPU acceleration:

- `nvidia-smi` should work and list your GPU
- `torch.cuda.is_available()` should be `True`
- `torch.version.cuda` should be non-empty (example: `11.8`)

If you see `+cpu` builds (for example `torch 2.x+cpu`) or `cuda False`, IDM-VTON is running on CPU.

### If your environment is CPU-only

Install CUDA-enabled PyTorch wheels in the `idm` environment (example for CUDA 11.8):

```bash
conda activate idm
python -m pip install --upgrade --force-reinstall \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

Then verify again:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Note: CUDA wheels require a compatible NVIDIA driver on the host machine.

### Installation workflow used in this project (Ubuntu 24.04 + RTX 3060)

The following sequence was executed and validated on this machine:

1. Detect hardware and recommended driver

```bash
lspci | grep -Ei 'vga|3d|nvidia'
ubuntu-drivers devices
```

2. Install NVIDIA driver (recommended profile)

```bash
sudo apt update
sudo apt install -y nvidia-driver-595-open
```

3. Verify driver stack

```bash
nvidia-smi
```

4. Free space before installing CUDA wheels (torch wheel is large)

```bash
conda clean -p -t -y
python -m pip cache purge
df -h /
```

5. Install CUDA-enabled torch stack in `idm`

```bash
conda activate idm
python -m pip install --upgrade --force-reinstall \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

6. Final verification

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -m pip check
```

Expected output on this setup:

- `torch 2.0.1+cu118`
- `torch.cuda.is_available() == True`
- `NVIDIA GeForce RTX 3060` detected

### Why disk appears small if the physical disk is 1TB

If `lsblk` shows an NVMe disk near 1TB but `df -h /` shows around 100G, the root filesystem is on an LVM logical volume with a limited size.

Example observed here:

- Physical disk: `nvme0n1` about 931G
- Volume group: `ubuntu-vg` about 929G total
- Root logical volume: `ubuntu-lv` only 100G
- Free space inside VG: about 829G not yet assigned to `/`

To verify your own layout:

```bash
df -h /
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT,TYPE
sudo vgs
sudo lvs
```

To expand root (advanced, run carefully):

```bash
sudo lvextend -l +100%FREE /dev/ubuntu-vg/ubuntu-lv
sudo resize2fs /dev/ubuntu-vg/ubuntu-lv
df -h /
```

This does not add a new disk; it grows the existing root filesystem to use free space already available in the same LVM volume group.

## Data preparation

### VITON-HD
You can download VITON-HD dataset from [VITON-HD](https://github.com/shadow2496/VITON-HD).

After download VITON-HD dataset, move vitonhd_test_tagged.json into the test folder, and move vitonhd_train_tagged.json into the train folder.

Structure of the Dataset directory should be as follows.

```

train
|-- image
|-- image-densepose
|-- agnostic-mask
|-- cloth
|-- vitonhd_train_tagged.json

test
|-- image
|-- image-densepose
|-- agnostic-mask
|-- cloth
|-- vitonhd_test_tagged.json

```

### DressCode
You can download DressCode dataset from [DressCode](https://github.com/aimagelab/dress-code).

We provide pre-computed densepose images and captions for garments [here](https://kaistackr-my.sharepoint.com/:u:/g/personal/cpis7_kaist_ac_kr/EaIPRG-aiRRIopz9i002FOwBDa-0-BHUKVZ7Ia5yAVVG3A?e=YxkAip).

We used [detectron2](https://github.com/facebookresearch/detectron2) for obtaining densepose images, refer [here](https://github.com/sangyun884/HR-VITON/issues/45) for more details.

After download the DressCode dataset, place image-densepose directories and caption text files as follows.

```
DressCode
|-- dresses
    |-- images
    |-- image-densepose
    |-- dc_caption.txt
    |-- ...
|-- lower_body
    |-- images
    |-- image-densepose
    |-- dc_caption.txt
    |-- ...
|-- upper_body
    |-- images
    |-- image-densepose
    |-- dc_caption.txt
    |-- ...
```


## Training


### Preparation

Download pre-trained ip-adapter for sdxl(IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin) and image encoder(IP-Adapter/models/image_encoder) [here](https://github.com/tencent-ailab/IP-Adapter).

```
git clone https://huggingface.co/h94/IP-Adapter
```

Move ip-adapter to ckpt/ip_adapter, and image encoder to ckpt/image_encoder.

Start training using python file with arguments,

```
accelerate launch train_xl.py \
    --gradient_checkpointing --use_8bit_adam \
    --output_dir=result --train_batch_size=6 \
    --data_dir=DATA_DIR
```

or, you can simply run with the script file.

```
sh train_xl.sh
```


## Inference


### VITON-HD

Inference using python file with arguments,

```
accelerate launch inference.py \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "result" \
    --unpaired \
    --data_dir "DATA_DIR" \
    --seed 42 \
    --test_batch_size 2 \
    --guidance_scale 2.0
```

or, you can simply run with the script file.

```
sh inference.sh
```

### DressCode

For DressCode dataset, put the category you want to generate images via category argument,
```
accelerate launch inference_dc.py \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "result" \
    --unpaired \
    --data_dir "DATA_DIR" \
    --seed 42 
    --test_batch_size 2
    --guidance_scale 2.0
    --category "upper_body" 
```

or, you can simply run with the script file.
```
sh inference.sh
```

## Start a local gradio demo <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a>

Download checkpoints for human parsing [here](https://huggingface.co/spaces/yisol/IDM-VTON/tree/main/ckpt).

Place the checkpoints under the ckpt folder.
```
ckpt
|-- densepose
    |-- model_final_162be9.pkl
|-- humanparsing
    |-- parsing_atr.onnx
    |-- parsing_lip.onnx

|-- openpose
    |-- ckpts
        |-- body_pose_model.pth
    
```




Run the following command:

```bash
IDMVTON_LOW_RAM=1 IDMVTON_SHARE=0 python gradio_demo/app.py
```

Runtime flags:

- `IDMVTON_LOW_RAM=1`: lower peak RAM usage (slower startup/inference)
- `IDMVTON_SHARE=0`: local-only mode (avoids public share tunnel)






## Acknowledgements


Thanks [ZeroGPU](https://huggingface.co/zero-gpu-explorers) for providing free GPU.

Thanks [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) for base codes.

Thanks [OOTDiffusion](https://github.com/levihsu/OOTDiffusion) and [DCI-VTON](https://github.com/bcmi/DCI-VTON-Virtual-Try-On) for masking generation.

Thanks [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) for human segmentation.

Thanks [Densepose](https://github.com/facebookresearch/DensePose) for human densepose.



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yisol/IDM-VTON&type=Date)](https://star-history.com/#yisol/IDM-VTON&Date)



## Citation
```
@article{choi2024improving,
  title={Improving Diffusion Models for Authentic Virtual Try-on in the Wild},
  author={Choi, Yisol and Kwak, Sangkyung and Lee, Kyungmin and Choi, Hyungwon and Shin, Jinwoo},
  journal={arXiv preprint arXiv:2403.05139},
  year={2024}
}
```



## License
The codes and checkpoints in this repository are under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).


