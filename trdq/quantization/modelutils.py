import gc
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
# from qdit.qBlock import QuantDiTBlock
from .gptq import GPTQ, Quantizer_GPTQ
from functools import partial

# from models.models import DiTBlock

from .quant import quantize_activation_wrapper, quantize_attn_v_wrapper, quantize_attn_k_wrapper, quantize_attn_q_wrapper
from .models.quant_layer import QuantLayer

def find_qlinear_layers(module, name=''):
    if type(module) == QuantLayer:  # 不需要判断fp层blocks不包含fp层
        # if module.weight_quant:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlinear_layers(
            child, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def quantize_model_gptq(model, device, args, dataloader):
    print('Starting GPTQ quantization ...')
    blocks = model.blocks

    quantizers = {}
    for i in tqdm(range(len(blocks))):
        m = blocks[i]

        block = m.to(device)

        block_layers = find_qlinear_layers(block)

        sequential = [list(block_layers.keys())]
       
        for names in sequential:
            subset = {n: block_layers[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer_GPTQ()
                gptq[name].quantizer.configure(
                    bits=4, perchannel=True, sym=False, mse=False,
                    channel_group=1,
                    clip_ratio=1.0,
                )
            # continue
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            model.to(device)
            for j in tqdm(range(31, 200, 32)):
                _ = model(dataloader[0][j-31:j+1], dataloader[1][j-31:j+1], dataloader[2][j-31:j+1], mask=dataloader[3][j-31:j+1])

            for h in handles:
                h.remove()
            
            for name in subset:
                gptq[name].fasterquant(
                    percdamp=0.01, groupsize=128, model=model, name=name, num=i
                )

                # subset[name].quantized = True
                quantizers['model.blocks.%d.%s' % (i, name)] = gptq[name].quantizer.cpu()
                gptq[name].free()


            del gptq

        # blocks[i] = block.cpu()  # 改需要这一步吗？?采取这一步后面代码变成cpu了
        del block, m
        torch.cuda.empty_cache()
        gc.collect()


    return model
