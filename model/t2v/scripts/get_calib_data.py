# Repository-local import bootstrap (keeps the original scripts directly executable).
from pathlib import Path as _Path
import sys as _sys
_SCRIPT_FILE = _Path(__file__).resolve()
_REPO_ROOT = _SCRIPT_FILE.parents[3]
_BACKEND_ROOT = _SCRIPT_FILE.parents[1]
for _local_path in (_REPO_ROOT, _BACKEND_ROOT):
    if str(_local_path) not in _sys.path:
        _sys.path.insert(0, str(_local_path))
import os

import torch
import colossalai
import torch.distributed as dist
from mmengine.runner import set_random_seed

from opensora.datasets import save_sample
from opensora.models.wan import resolve_wan_pretrained_root
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.build_model import build_wan_components
from opensora.utils.misc import to_torch_dtype
from opensora.acceleration.parallel_states import set_sequence_parallel_group
from colossalai.cluster import DistCoordinator


def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts


def main():
    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    cfg = parse_configs(training=False, mode="get_calib")
    print(cfg)

    # init distributed
    # colossalai.launch_from_torch({})
    # coordinator = DistCoordinator()

    # if coordinator.world_size > 1:
        # set_sequence_parallel_group(dist.group.WORLD) 
        # enable_sequence_parallelism = True
    # else:
        # enable_sequence_parallelism = False
    
    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.dtype)
    set_random_seed(seed=cfg.seed)
    prompts = load_prompts(cfg.prompt_path)
    prompts = prompts[:cfg.data_num]
    PRECOMPUTE_TEXT_EMBEDS = cfg.get('precompute_text_embeds', None)

    if cfg.model.get("model_type", None) == "wan":
        return run_wan_calibration(cfg, device, dtype, prompts)

    # ======================================================
    # 3. build model & load weights
    # =====================================
    # 
    # =================
    # 3.1. build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    
    # 3.2. build model
    input_size = (cfg.num_frames, *cfg.image_size)
    vae = build_module(cfg.vae, MODELS)
    latent_size = vae.get_latent_size(input_size)
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        # caption_channels=text_encoder.output_dim,
        caption_channels=4096,  # DIRTY: for T5 only
        model_max_length=cfg.text_encoder.model_max_length,
        dtype=dtype,
        enable_sequence_parallelism=False,
    )
    if PRECOMPUTE_TEXT_EMBEDS is not None:
        text_encoder = None
    else:
        text_encoder = build_module(cfg.text_encoder, MODELS, device=device)  # T5 must be fp32
        text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance

    # 3.3. move to device & eval
    vae = vae.to(device, dtype).eval()
    model = model.to(device, dtype).eval()

    # 3.4. support for multi-resolution
    model_args = dict()
    if cfg.multi_resolution:
        image_size = cfg.image_size
        hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        model_args["data_info"] = dict(ar=ar, hw=hw)

    # ======================================================
    # 4. inference
    # ======================================================
    sample_idx = 0
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    calib_data = {}
    input_data_list = []
    output_data_list = []
    if PRECOMPUTE_TEXT_EMBEDS is not None:
        model_args['precompute_text_embeds'] = torch.load(cfg.precompute_text_embeds)

    for i in range(0, len(prompts), cfg.batch_size):
        batch_prompts = prompts[i : i + cfg.batch_size]
        if PRECOMPUTE_TEXT_EMBEDS is not None:  # also feed in the idxs for saved text_embeds
            model_args['batch_ids'] = torch.arange(i,i+cfg.batch_size)
        samples, cur_calib_data, out_data = scheduler.sample(
            model,
            text_encoder,
            sampler_type=cfg.sampler,
            z_size=(vae.out_channels, *latent_size),
            prompts=batch_prompts,
            device=device,
            return_trajectory=True,  # 返回轨迹
            additional_args=model_args,
        )

        for key in cur_calib_data:
            if not key in calib_data.keys():
                calib_data[key] = cur_calib_data[key]
            else:
                calib_data[key] = torch.cat([cur_calib_data[key], calib_data[key]], dim=1)

        input_data_list.append(cur_calib_data)
        output_data_list.append(out_data)
        # save samples
        samples = vae.decode(samples.to(dtype))
        # if coordinator.is_master():
        for idx, sample in enumerate(samples):
            print(f"Prompt: {batch_prompts[idx]}")
            save_path = os.path.join(save_dir, f"sample_{sample_idx}")
            save_sample(sample, fps=cfg.fps, save_path=save_path)
            sample_idx += 1

    # ======================================================
    # 5. save calibration data
    # ======================================================
    torch.save(calib_data, os.path.join(save_dir, "calib_data.pt"))
    if cfg.save_inp_oup:
        torch.save(input_data_list, os.path.join(save_dir, "input_list.pt"))
        torch.save(output_data_list, os.path.join(save_dir, "output_list.pt"))


def run_wan_calibration(cfg, device, dtype, prompts):
    try:
        from diffusers import WanPipeline
        from diffusers.utils import export_to_video
    except ImportError as exc:
        print(f"Warning: {exc}")
        print("Wan calibration requires recent `diffusers`, but continuing with basic calibration...")
        # Create a dummy calibration data file
        save_dir = cfg.save_dir
        os.makedirs(save_dir, exist_ok=True)
        calib_data = {"xs": [], "ts": [], "cond_emb": [], "mask": []}
        torch.save(calib_data, os.path.join(save_dir, "calib_data.pt"))
        print(f"Created dummy calibration data at {os.path.join(save_dir, 'calib_data.pt')}")
        return

    _, model, text_encoder, vae, _, _ = build_wan_components(cfg, device, dtype)
    pipeline_root = resolve_wan_pretrained_root(cfg.pipeline_from_pretrained)
    pipe = WanPipeline.from_pretrained(
        pipeline_root,
        transformer=model.model,
        vae=vae.model,
        text_encoder=text_encoder.text_encoder,
        tokenizer=text_encoder.tokenizer,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    calib_data = {"xs": [], "ts": [], "cond_emb": [], "mask": []}
    sample_idx = 0

    for i in range(0, len(prompts), cfg.batch_size):
        batch_prompts = prompts[i : i + cfg.batch_size]
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=batch_prompts,
            negative_prompt=[""] * len(batch_prompts),
            do_classifier_free_guidance=cfg.scheduler.get("cfg_scale", 1.0) > 1.0,
            num_videos_per_prompt=1,
            device=device,
            max_sequence_length=cfg.text_encoder.model_max_length,
        )
        prompt_mask = (prompt_embeds.abs().sum(dim=-1) > 0).to(torch.int64)

        def _capture_callback(_pipe, step_idx, timestep, callback_kwargs):
            latents = callback_kwargs["latents"].detach().cpu()
            ts = torch.full((latents.shape[0],), int(timestep), dtype=torch.long)
            calib_data["xs"].append(latents)
            calib_data["ts"].append(ts)
            calib_data["cond_emb"].append(prompt_embeds.detach().cpu())
            calib_data["mask"].append(prompt_mask.detach().cpu())
            return callback_kwargs

        videos = pipe(
            prompt=batch_prompts,
            negative_prompt=[""] * len(batch_prompts),
            num_frames=cfg.num_frames,
            height=cfg.image_size[0],
            width=cfg.image_size[1],
            guidance_scale=cfg.scheduler.get("cfg_scale", 1.0),
            num_inference_steps=cfg.scheduler.get("num_sampling_steps", cfg.scheduler.get("num_inference_steps", 50)),
            callback_on_step_end=_capture_callback,
            callback_on_step_end_tensor_inputs=["latents"],
            output_type="np",
        ).frames

        for video in videos:
            save_path = os.path.join(save_dir, f"sample_{sample_idx}.mp4")
            export_to_video(video, save_path, fps=cfg.fps)
            sample_idx += 1

    torch.save(calib_data, os.path.join(save_dir, "calib_data.pt"))


if __name__ == "__main__":
    main()
