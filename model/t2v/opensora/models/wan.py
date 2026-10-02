import math
import os
from typing import Optional

import torch
import torch.nn as nn

from opensora.registry import MODELS


def _require_wan_backends():
    try:
        from diffusers import AutoencoderKLWan, WanTransformer3DModel
        from transformers import AutoTokenizer, UMT5EncoderModel
    except ImportError as exc:
        raise ImportError(
            "Wan support requires recent `diffusers` and `transformers` packages. "
            "Install the updated requirements before using Wan configs."
        ) from exc

    return AutoencoderKLWan, WanTransformer3DModel, AutoTokenizer, UMT5EncoderModel


def _to_torch_dtype(dtype):
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        dtype_map = {
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype.lower(), None)
    return None


def _is_wan_diffusers_root(path: str) -> bool:
    required_dirs = ("transformer", "vae", "text_encoder", "tokenizer")
    return os.path.isdir(path) and all(os.path.isdir(os.path.join(path, name)) for name in required_dirs)


def _is_wan_raw_root(path: str) -> bool:
    required_dirs = ("high_noise_model", "low_noise_model", "google")
    required_files = ("configuration.json", "models_t5_umt5-xxl-enc-bf16.pth")
    return os.path.isdir(path) and all(os.path.isdir(os.path.join(path, name)) for name in required_dirs) and all(
        os.path.exists(os.path.join(path, name)) for name in required_files
    )


def resolve_wan_pretrained_root(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None

    if not os.path.isdir(path):
        return path

    resolved = os.path.abspath(path)
    if _is_wan_diffusers_root(resolved):
        return resolved

    if _is_wan_raw_root(resolved):
        raise ValueError(
            "Detected a local Wan raw checkpoint layout at "
            f"`{resolved}`, but the current TR-DQ Wan adapter expects a Diffusers-style checkpoint root "
            "with `transformer/`, `vae/`, `text_encoder/`, and `tokenizer/` subdirectories. "
            "Your raw Wan assets have been organized for reference, but they are not directly loadable by the "
            "current Diffusers-based PTQ pipeline. Point `CKPT_PATH` to a local Diffusers export such as "
            "`.../pretrained_models/wan/Wan2.2-T2V-A14B-Diffusers`."
        )

    return resolved


@MODELS.register_module("wan_transformer")
@MODELS.register_module("WanTransformer3DModel")
class WanTransformer3DModelWrapper(nn.Module):
    def __init__(
        self,
        from_pretrained: Optional[str] = None,
        subfolder: str = "transformer",
        dtype=None,
        patch_size=(1, 2, 2),
        num_attention_heads=40,
        attention_head_dim=128,
        in_channels=16,
        out_channels=16,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=13824,
        num_layers=40,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        image_dim=None,
        added_kv_proj_dim=None,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        **kwargs,
    ):
        super().__init__()
        _, WanTransformer3DModel, _, _ = _require_wan_backends()

        torch_dtype = _to_torch_dtype(dtype)
        if from_pretrained is not None:
            from_pretrained = resolve_wan_pretrained_root(from_pretrained)
            self.model = WanTransformer3DModel.from_pretrained(
                from_pretrained,
                subfolder=subfolder,
                torch_dtype=torch_dtype,
            )
        else:
            self.model = WanTransformer3DModel(
                patch_size=patch_size,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                in_channels=in_channels,
                out_channels=out_channels,
                text_dim=text_dim,
                freq_dim=freq_dim,
                ffn_dim=ffn_dim,
                num_layers=num_layers,
                cross_attn_norm=cross_attn_norm,
                qk_norm=qk_norm,
                eps=eps,
                image_dim=image_dim,
                added_kv_proj_dim=added_kv_proj_dim,
                rope_max_seq_len=rope_max_seq_len,
                pos_embed_seq_len=pos_embed_seq_len,
            )

        self.config = self.model.config
        self.blocks = self.model.blocks
        self.in_channels = self.model.config.in_channels
        self.out_channels = self.model.config.out_channels

    def forward(
        self,
        x,
        t,
        y,
        attention_mask=None,
        encoder_hidden_states_image=None,
        return_dict=True,
        **kwargs,
    ):
        output = self.model(
            hidden_states=x,
            timestep=t,
            encoder_hidden_states=y,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            **kwargs,
        )
        if return_dict:
            return {"x": output.sample}
        return output[0]


@MODELS.register_module("wan_vae")
@MODELS.register_module("AutoencoderKLWan")
class AutoencoderKLWanWrapper(nn.Module):
    def __init__(
        self,
        from_pretrained: Optional[str] = None,
        subfolder: str = "vae",
        dtype=None,
        scale_factor_temporal=4,
        scale_factor_spatial=8,
        z_dim=16,
        **kwargs,
    ):
        super().__init__()
        AutoencoderKLWan, _, _, _ = _require_wan_backends()

        torch_dtype = _to_torch_dtype(dtype)
        if from_pretrained is not None:
            from_pretrained = resolve_wan_pretrained_root(from_pretrained)
            self.model = AutoencoderKLWan.from_pretrained(
                from_pretrained,
                subfolder=subfolder,
                torch_dtype=torch_dtype,
            )
        else:
            self.model = AutoencoderKLWan(
                scale_factor_temporal=scale_factor_temporal,
                scale_factor_spatial=scale_factor_spatial,
                z_dim=z_dim,
                **kwargs,
            )

        self.out_channels = self.model.config.z_dim
        self.scale_factor_temporal = getattr(self.model.config, "scale_factor_temporal", scale_factor_temporal)
        self.scale_factor_spatial = getattr(self.model.config, "scale_factor_spatial", scale_factor_spatial)

    def get_latent_size(self, input_size):
        num_frames, height, width = input_size
        latent_frames = (num_frames - 1) // self.scale_factor_temporal + 1
        latent_height = math.ceil(height / self.scale_factor_spatial)
        latent_width = math.ceil(width / self.scale_factor_spatial)
        return latent_frames, latent_height, latent_width

    def encode(self, x):
        posterior = self.model.encode(x).latent_dist
        return posterior.sample()

    def decode(self, z):
        return self.model.decode(z).sample

    def forward(self, x):
        return self.model(x)


@MODELS.register_module("wan_t5")
@MODELS.register_module("WanT5Encoder")
class WanT5Encoder:
    def __init__(
        self,
        from_pretrained: Optional[str] = None,
        text_encoder_subfolder: str = "text_encoder",
        tokenizer_subfolder: str = "tokenizer",
        model_max_length: int = 512,
        device: str = "cuda",
        dtype=torch.float32,
        **kwargs,
    ):
        _, _, AutoTokenizer, UMT5EncoderModel = _require_wan_backends()

        if from_pretrained is None:
            raise ValueError("WanT5Encoder requires `from_pretrained` to point to a Wan checkpoint directory.")

        from_pretrained = resolve_wan_pretrained_root(from_pretrained)
        torch_dtype = _to_torch_dtype(dtype) or torch.float32
        self.device = device
        self.model_max_length = model_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(from_pretrained, subfolder=tokenizer_subfolder)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            from_pretrained,
            subfolder=text_encoder_subfolder,
            torch_dtype=torch_dtype,
        ).to(device)
        self.text_encoder.eval()
        self.output_dim = self.text_encoder.config.d_model

    @torch.no_grad()
    def encode(self, text):
        if isinstance(text, str):
            text = [text]
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.model_max_length,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        hidden_states = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return dict(y=hidden_states, mask=attention_mask)

    @torch.no_grad()
    def null(self, n):
        return self.encode([""] * n)["y"]
