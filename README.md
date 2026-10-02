# TR-DQ: Time-Rotation Diffusion Quantization (Accepted by the Fortieth AAAI Conference on Artificial Intelligence, AAAI-26)

<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h5>

<p align="center">
  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/37841"><img src="https://img.shields.io/badge/AAAI%202026-Paper-red" alt="AAAI 2026 Paper"></a>
  <a href="https://arxiv.org/abs/2503.06564"><img src="https://img.shields.io/badge/arXiv-2503.06564-b31b1b" alt="arXiv"></a>
  <a href="README_zh-CN.md"><img src="https://img.shields.io/badge/README-中文-blue" alt="中文版 README"></a>
</p>

## News

- **[2025.11.08]** Our paper was accepted by the Fortieth AAAI Conference on Artificial Intelligence, AAAI-26.
- Code will be released soon.

<p align="center">
  <img src="assets/trdq_overview.png" alt="TR-DQ method overview" width="95%">
</p>

## Requirements

- Linux, Python 3.10
- PyTorch 2.3.1, CUDA 12.1
- NVIDIA GPU; the paper experiments used A800 GPUs

PixArt and OpenSora require different Diffusers versions, so separate environments are recommended.

```bash
# PixArt
conda create -n trdq-image python=3.10 -y
conda activate trdq-image
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/image.txt && pip install -e . --no-deps

# OpenSora
conda create -n trdq-video python=3.10 -y
conda activate trdq-video
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/video.txt && pip install -e . --no-deps
```

## Pretrained Models

| Model | Official download | Local path |
| --- | --- | --- |
| PixArt-alpha | [PixArt-alpha/PixArt-alpha](https://huggingface.co/PixArt-alpha/PixArt-alpha) | `checkpoints/pixart_alpha/` |
| OpenSora v1 HQ | [OpenSora-v1-HQ-16x512x512.pth](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x512x512.pth) | `checkpoints/OpenSora-v1-HQ-16x512x512.pth` |
| SD VAE | [stabilityai/sd-vae-ft-ema](https://huggingface.co/stabilityai/sd-vae-ft-ema) | `checkpoints/sd-vae-ft-ema/` |
| T5 | [DeepFloyd/t5-v1_1-xxl](https://huggingface.co/DeepFloyd/t5-v1_1-xxl) | `checkpoints/t5-v1_1-xxl/` |

## Project Structure

```text
.
├── assets/                 # Method overview figure
├── config/                 # W4A8 and W8A8 experiment configurations
├── model/
│   ├── t2i/                # PixArt calibration, PTQ, and inference
│   └── t2v/                # OpenSora calibration, PTQ, and inference
├── requirements/           # Image and video environment dependencies
├── test/                   # Lightweight CPU tests
├── trdq/quantization/      # TR-DQ quantizers and reconstruction modules
└── utils/                  # Shared utilities
```

## Usage

Run all commands from the repository root.

### PixArt-alpha W4A8

```bash
# 1. Calibration
python model/t2i/scripts/get_calib_data.py \
  --version alpha --txt_file model/t2i/asset/calib.txt \
  --pipeline_load_from checkpoints/pixart_alpha \
  --model_path checkpoints/pixart_alpha/PixArt-XL-2-1024-MS.pth \
  --save_path outputs/pixart_alpha/calibration --bs 8

# 2. PTQ
python model/t2i/scripts/ptq.py \
  --version alpha --txt_file model/t2i/asset/calib.txt \
  --pipeline_load_from checkpoints/pixart_alpha \
  --model_path checkpoints/pixart_alpha/PixArt-XL-2-1024-MS.pth \
  --ptq_config config/pixart_alpha/w4a8.yaml \
  --calib_data_path outputs/pixart_alpha/calibration \
  --save_path outputs/pixart_alpha --exp_name w4a8

# 3. Inference
python model/t2i/scripts/quant_txt2img.py \
  --version alpha --txt_file model/t2i/asset/coco_1024.txt \
  --pipeline_load_from checkpoints/pixart_alpha \
  --model_path checkpoints/pixart_alpha/PixArt-XL-2-1024-MS.pth \
  --quant_path outputs/pixart_alpha/alpha/w4a8 \
  --save_path outputs/pixart_alpha --quant_weight --quant_act --bs 8
```

### OpenSora W4A8

Split the fused QKV checkpoint once:

```bash
python model/t2v/scripts/split_ckpt.py \
  checkpoints/OpenSora-v1-HQ-16x512x512.pth \
  checkpoints/OpenSora-v1-HQ-16x512x512-split.pth
```

```bash
# 1. Calibration
python model/t2v/scripts/get_calib_data.py config/opensora/model.py \
  --ckpt_path checkpoints/OpenSora-v1-HQ-16x512x512-split.pth \
  --outdir outputs/opensora/calibration \
  --save_dir outputs/opensora/calibration --data_num 10 --gpu 0

# 2. PTQ
python model/t2v/scripts/ptq.py config/opensora/model.py \
  --ckpt_path checkpoints/OpenSora-v1-HQ-16x512x512-split.pth \
  --ptq_config config/opensora/w4a8.yaml \
  --calib_data outputs/opensora/calibration/calib_data.pt \
  --outdir outputs/opensora/w4a8 --quant_ckpt_name ckpt.pth --part_fp --gpu 0

# 3. Inference
python model/t2v/scripts/quant_txt2video.py config/opensora/model.py \
  --ckpt_path checkpoints/OpenSora-v1-HQ-16x512x512-split.pth \
  --ptq_config config/opensora/w4a8.yaml \
  --quant_ckpt outputs/opensora/w4a8/ckpt.pth \
  --prompt_path model/t2v/assets/texts/t2v_samples_10.txt \
  --outdir outputs/opensora/w4a8 --save_dir outputs/opensora/w4a8/videos \
  --dataset_type opensora --part_fp --gpu 0
```

Add `--attention_sharing` to the last command for the TR-DQ+AS setting. W8A8 configurations are also provided under `config/`.

## Citation

```bibtex
@inproceedings{shao2026trdq,
  title={TR-DQ: Time-Rotation Diffusion Quantization},
  author={Shao, Yihua and Lin, Deyang and Yan, Minxi and Chen, Siyu and Zeng, Fanhu and Liao, Minwen and Ma, Ao and Yan, Ziyang and Wang, Haozhe and Wang, Yan and Chen, Zhi and Cao, Xiaofeng and Qin, Haotong and Tang, Hao and Guo, Jingcai},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40}, number={11}, pages={8869--8877}, year={2026}
}
```

## Acknowledgements

We thank the authors of the following repositories and papers for their excellent work.

## References

### Repositories

- [Q-Diffusion](https://github.com/Xiuyu-Li/q-diffusion)
- [ViDiT-Q](https://github.com/thu-nics/ViDiT-Q)
- [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha)
- [Open-Sora](https://github.com/hpcaitech/Open-Sora)

### Papers

- [Q-Diffusion: Quantizing Diffusion Models](https://arxiv.org/abs/2302.04304)
- [ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation](https://arxiv.org/abs/2406.02540)
- [PixArt-alpha: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis](https://arxiv.org/abs/2310.00426)
- [Open-Sora: Democratizing Efficient Video Production for All](https://arxiv.org/abs/2412.20404)
