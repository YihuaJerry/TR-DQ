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
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
import sys
# sys.path.append(".")

import torch
import shutil
import logging
from omegaconf import OmegaConf
from mmengine.runner import set_random_seed

from opensora.datasets import save_sample
from opensora.models.wan import resolve_wan_pretrained_root
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.build_model import build_models, build_wan_components
from opensora.utils.misc import to_torch_dtype

from trdq.quantization.models.quant_model import QuantModel
from trdq.quantization.utils import load_quant_params

import inspect
import os

def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts


def main():
    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    cfg = parse_configs(training=False, mode="quant_inference")
    print(cfg)
    PRECOMPUTE_TEXT_EMBEDS = cfg.get('precompute_text_embeds', None)

    opt = cfg
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    
    # INFO: add bakup file and bakup cfg into logpath for debug
    # load the config from the log path
    if not opt.get("ptq_config"):
        opt.ptq_config = "./config/opensora/w4a8.yaml"
    if not opt.get("quant_ckpt") or not os.path.exists(opt.quant_ckpt):
        opt.quant_ckpt = os.path.join(opt.outdir, opt.get("ckptt", "ckpt.pth"))
    if not hasattr(opt,"save_dir"):
        opt.save_dir = os.path.join(opt.outdir,'generated_videos')
    config = OmegaConf.load(f"{opt.ptq_config}")

    log_path = os.path.join(outpath, "quant_inference_run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Conducting Command: %s", " ".join(sys.argv))

    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    gpus = [int(d) for d in cfg.gpu.split(",")]
    torch.cuda.set_device(gpus[0])
    torch.cuda.empty_cache()
    print("Current CUDA device:", torch.cuda.current_device())
    dtype = to_torch_dtype(cfg.dtype)
    set_random_seed(seed=cfg.seed)
    prompts = load_prompts(cfg.prompt_path)
    prompts = prompts[:cfg.num_videos]
    config = OmegaConf.load(f"{opt.ptq_config}")

    # ======================================================
    # 3. build model & load weights
    # ======================================================
    if cfg.model.get("model_type", None) == "wan":
        scheduler, model, text_encoder, vae, model_args, latent_size = build_wan_components(cfg, device, dtype)
    else:
        scheduler = build_module(cfg.scheduler, SCHEDULERS)
        if str(cfg.get("atnn", "False")).lower() == "true":
            cfg.model.enable_flashattn = True
        else:
            cfg.model.enable_flashattn = False
        cfg.model.attention_sharing = bool(cfg.get("attention_sharing", False))

        input_size = (cfg.num_frames, *cfg.image_size)
        vae = build_module(cfg.vae, MODELS)
        latent_size = vae.get_latent_size(input_size)
        model = build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels,
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

        vae = vae.to(device, dtype).eval()
        model = model.to(device, dtype).eval()

        model_args = dict()
        if cfg.multi_resolution:
            image_size = cfg.image_size
            hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
            ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
            model_args["data_info"] = dict(ar=ar, hw=hw)


    # scheduler, model, text_encoder, vae, model_args, latent_size = build_models(cfg, device, dtype, enable_sequence_parallelism=False)
    # assert(config.conditional)
    # ======================================================
    # 4. get quantized model
    # ======================================================
    num_timesteps = config.calib_data.n_steps

    assert(config.conditional)

    wq_params = config.quant.weight.quantizer
    aq_params = config.quant.activation.quantizer
    use_weight_quant = True if wq_params else False
    use_act_quant = True if aq_params else False
    if opt.skip_quant_weight:
        use_weight_quant = False
    if opt.skip_quant_act:
        use_act_quant = False

    if config.get('mixed_precision', False):
        if use_weight_quant:
            wq_params['mixed_precision'] = config.mixed_precision
        # if use_act_quant:
        #     aq_params['mixed_precision'] = config.mixed_precision

    qnn = QuantModel(
        model=model, \
        weight_quant_params=wq_params,\
        act_quant_params=aq_params,\
        model_type=config.model.model_type,
    )
    qnn.cuda()
    qnn.eval()
    logger.info(qnn)

    # DIRTY: set the cfg_split as the attribute of the model
    # the cfg_split is configured in `opensora/schedulers/ippdm/__init__.py`
    cfg_split = config.get('cfg_split', False)
    qnn.cfg_split = cfg_split
    
    qnn.set_quant_state(False, False)
    # for smooth quant
    qnn.set_smooth_quant(smooth_quant=False, smooth_quant_running_stat=False)
    calib_added_cond = {} # It is not required for STDiT

        # if "opensora" in config.model.model_id:
            # _ = qnn(torch.randn(1, 4, 16, 64, 64).cuda(), torch.randint(0, 1000, (1,)).cuda(), torch.randn(1, 1, 120, 4096).cuda(), mask=torch.ones(1, 120).cuda().to(torch.int64))
        # else:
            # raise NotImplementedError

    # for part quantization
    if opt.part_quant:
        quant_layer_list = list(torch.load(config.part_quant_list))
        quant_layer_list = quant_layer_list[:int(len(quant_layer_list) * opt.quant_ratio)]

    if opt.part_fp:
        with open(config.part_fp_list,'r') as f:
            lines = f.readlines()
        fp_layer_list = [line.strip() for line in lines]  # strip the '\n'
        if opt.get('fp_ratio',None) is not None:
            fp_layer_list = fp_layer_list[:int(len(fp_layer_list) * opt.fp_ratio)]
        logger.info("Set the following layers as FP: {}".format(fp_layer_list))

    # for smooth quant
    if aq_params.smooth_quant.enable:
        qnn.set_smooth_quant(smooth_quant=False, smooth_quant_running_stat=False)
        # for i in range(len(sens)):
        #     if sens[i][1]["fp16_diff"] > 7.0 or sens[i][1]["fp16_diff"] < 0.5:
        #         smooth_quant_layer_list.append(sens[i][0])
                # alpha_dict[sens[i][0]] = sens[i][1]["best_alpha"]
        # alpha_dict["model.blocks.27.mlp.fc2"] = 0.675
        qnn.set_smooth_quant(smooth_quant=True, smooth_quant_running_stat=False) # Now we use fp16 to save the statistic of activation
        qnn.set_layer_smooth_quant(model=qnn, module_name_list=fp_layer_list, smooth_quant=False, smooth_quant_running_stat=False)
        # qnn.set_layer_smooth_quant_alpha(model=qnn, alpha_dict=alpha_dict)

    # set the init flag True, otherwise will recalculate params
    if opt.part_quant:
        qnn.set_layer_quant(model=qnn, module_name_list=quant_layer_list, quant_level='per_layer', weight_quant=use_weight_quant, act_quant=use_act_quant, prefix="")
    elif opt.part_fp:
        qnn.set_quant_state(use_weight_quant, use_act_quant)
        qnn.set_layer_quant(model=qnn, module_name_list=fp_layer_list, quant_level='per_layer', weight_quant=False, act_quant=False, prefix="")
    else:
        qnn.set_quant_state(use_weight_quant, use_act_quant) # enable weight quantization, disable act quantization
    qnn.set_quant_init_done('weight')
    qnn.set_quant_init_done('activation')
    load_quant_params(qnn, opt.quant_ckpt)
    qnn.cuda()
    # duquant
    float32_params = {}

    def save_quantizer_permutations(module, prefix):
        """保存模块中act_quantizer和weight_quantizer的permutation_list"""
        # 保存act_quantizer
        if hasattr(module, 'act_quantizer'):
            quantizer = module.act_quantizer
            if hasattr(quantizer, 'permutation_list') and quantizer.permutation_list is not None:
                float32_params[f'{prefix}_act_perm'] = quantizer.permutation_list.clone()

        # 保存weight_quantizer
        if hasattr(module, 'weight_quantizer'):
            quantizer = module.weight_quantizer
            if hasattr(quantizer, 'permutation_list') and quantizer.permutation_list is not None:
                float32_params[f'{prefix}_weight_perm'] = quantizer.permutation_list.clone()

    def save_attn_permutations(attn_block, prefix):
        """保存注意力模块中的permutation_list"""
        for name in ['q', 'k', 'v', 'proj']:
            if hasattr(attn_block, name):
                module = getattr(attn_block, name)
                save_quantizer_permutations(module, f'{prefix}_{name}')

    # 遍历所有blocks
    for i in range(len(qnn.model.blocks)):
        block = qnn.model.blocks[i]

        # 空间注意力
        if hasattr(block, 'attn'):
            save_attn_permutations(block.attn, f'block_{i}_spatial_attn')

        # 交叉注意力
        if hasattr(block, 'cross_attn'):
            for name in ['q_linear', 'kv_linear', 'proj']:
                if hasattr(block.cross_attn, name):
                    module = getattr(block.cross_attn, name)
                    save_quantizer_permutations(module, f'block_{i}_cross_attn_{name}')

        # 时序注意力
        if hasattr(block, 'attn_temp'):
            save_attn_permutations(block.attn_temp, f'block_{i}_temporal_attn')

        # MLP
        if hasattr(block, 'mlp'):
            for name in ['fc1', 'fc2']:
                if hasattr(block.mlp, name):
                    module = getattr(block.mlp, name)
                    save_quantizer_permutations(module, f'block_{i}_mlp_{name}')

    # 3. 转换模型精度
    qnn.to(dtype)

    # 4. 恢复所有permutation_list
    def restore_quantizer_permutations(module, prefix):
        """恢复模块中act_quantizer和weight_quantizer的permutation_list"""
        # 恢复act_quantizer
        if hasattr(module, 'act_quantizer'):
            key = f'{prefix}_act_perm'
            if key in float32_params:
                module.act_quantizer.permutation_list = float32_params[key]
        # 恢复weight_quantizer
        if hasattr(module, 'weight_quantizer'):
            key = f'{prefix}_weight_perm'
            if key in float32_params:
                module.weight_quantizer.permutation_list = float32_params[key]
    def restore_attn_permutations(attn_block, prefix):
        """恢复注意力模块中的permutation_list"""
        for name in ['q', 'k', 'v', 'proj']:
            if hasattr(attn_block, name):
                module = getattr(attn_block, name)
                restore_quantizer_permutations(module, f'{prefix}_{name}')
    for i in range(len(qnn.model.blocks)):
        block = qnn.model.blocks[i]
        # 恢复空间注意力
        if hasattr(block, 'attn'):
            restore_attn_permutations(block.attn, f'block_{i}_spatial_attn')
        # 恢复交叉注意力
        if hasattr(block, 'cross_attn'):
            for name in ['q_linear', 'kv_linear', 'proj']:
                if hasattr(block.cross_attn, name):
                    module = getattr(block.cross_attn, name)
                    restore_quantizer_permutations(module, f'block_{i}_cross_attn_{name}')
        # 恢复时序注意力
        if hasattr(block, 'attn_temp'):
            restore_attn_permutations(block.attn_temp, f'block_{i}_temporal_attn')
        # 恢复MLP
        if hasattr(block, 'mlp'):
            for name in ['fc1', 'fc2']:
                if hasattr(block.mlp, name):
                    module = getattr(block.mlp, name)
                    restore_quantizer_permutations(module, f'block_{i}_mlp_{name}')
    #duquant
    # qnn.cuda()
    # qnn.to(dtype)

    # ======================================================
    # 5. inference
    # ======================================================
    qnn.timestep_wise_quant =False

    sample_idx = 0
    save_dir = opt.save_dir
    save_dir = os.path.join(save_dir, 'generated_videos')
    os.makedirs(save_dir, exist_ok=True)
    if cfg.model.get("model_type", None) == "wan":
        run_wan_quant_inference(cfg, dtype, prompts, qnn, text_encoder, vae, opt.save_dir)
        return

    if PRECOMPUTE_TEXT_EMBEDS is not None:
        model_args['precompute_text_embeds'] = torch.load(cfg.precompute_text_embeds, map_location="cuda:" + str(torch.cuda.current_device()))  # 转到对应的gpu去
    for i in range(0, len(prompts), 4):
        batch_prompts = prompts[i:i+4]
        if PRECOMPUTE_TEXT_EMBEDS is not None:  # also feed in the idxs for saved text_embeds
            model_args['batch_ids'] = torch.arange(i,i+4)
        samples = scheduler.sample(
            qnn,
            text_encoder,
            sampler_type=cfg.sampler,
            z_size=(vae.out_channels, *latent_size),
            prompts=batch_prompts,
            device=device,
            additional_args=model_args,
        )
        samples = vae.decode(samples.to(dtype))

        for idx, sample in enumerate(samples):
            print(f"Prompt: {batch_prompts[idx]}")
            dirs = cfg.get("dirss", None) or cfg.save_dir
            # dirs = "./logs/video" + "/overall/20_w4a8_attn"
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            save_path = os.path.join(dirs, f"sample_{sample_idx}")
            save_sample(sample, fps=cfg.fps, save_path=save_path)
            sample_idx += 1


def run_wan_quant_inference(cfg, dtype, prompts, qnn, text_encoder, vae, save_dir):
    try:
        from diffusers import WanPipeline
        from diffusers.utils import export_to_video
    except ImportError as exc:
        raise ImportError("Wan quantized inference requires recent `diffusers`.") from exc

    class QuantizedWanTransformerAdapter(torch.nn.Module):
        def __init__(self, quant_model):
            super().__init__()
            self.quant_model = quant_model
            self.config = quant_model.model.config
            self.blocks = quant_model.model.blocks

        def forward(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image=None,
            return_dict=True,
            **kwargs,
        ):
            return self.quant_model(
                hidden_states,
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
                return_dict=return_dict,
                **kwargs,
            )

    pipeline_root = resolve_wan_pretrained_root(cfg.pipeline_from_pretrained)
    pipe = WanPipeline.from_pretrained(
        pipeline_root,
        transformer=QuantizedWanTransformerAdapter(qnn),
        vae=vae.model,
        text_encoder=text_encoder.text_encoder,
        tokenizer=text_encoder.tokenizer,
        torch_dtype=dtype,
    )
    pipe = pipe.to(next(qnn.parameters()).device)

    save_dir = os.path.join(save_dir, "generated_videos")
    os.makedirs(save_dir, exist_ok=True)
    for i in range(0, len(prompts), cfg.batch_size):
        batch_prompts = prompts[i : i + cfg.batch_size]
        output = pipe(
            prompt=batch_prompts,
            negative_prompt=[""] * len(batch_prompts),
            num_frames=cfg.num_frames,
            height=cfg.image_size[0],
            width=cfg.image_size[1],
            guidance_scale=cfg.scheduler.get("cfg_scale", 1.0),
            num_inference_steps=cfg.scheduler.get("num_sampling_steps", cfg.scheduler.get("num_inference_steps", 50)),
            output_type="np",
        )
        for local_idx, video in enumerate(output.frames):
            export_to_video(video, os.path.join(save_dir, f"sample_{i + local_idx}.mp4"), fps=cfg.fps)


if __name__ == "__main__":
    main()
