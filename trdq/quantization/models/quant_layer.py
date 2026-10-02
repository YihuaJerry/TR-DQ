import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import time  # DEBUG_ONLY
from omegaconf import ListConfig
import copy
from trdq.quantization.quantizer.base_quantizer import WeightQuantizer, ActQuantizer, StraightThrough
from trdq.quantization.quantizer.dynamic_quantizer import DynamicActQuantizer
import diffusers

logger = logging.getLogger(__name__)

def find_interval(timerange, timestep_id):
    # timestep_id = int(timestep_id*1000)
    for index, interval in enumerate(timerange):
        if interval[0] <= timestep_id <= interval[1]:
            return index
    return None  # If timestep_id is not within any interval
# duquant
import numpy as np
import matplotlib.pyplot as plt
# from diffusion.model.nets import glo

block = 0
def save_3d_distributions(input, save_dir='./logs/figure/', name=None):
    # 确保保存目录存在
    import os
    from einops import rearrange
    os.makedirs(save_dir, exist_ok=True)
    data = input.detach().clone()
    if (len(input.shape) == 3):
        tmp = data.shape[1]
        tensor = rearrange(data, "B (T S) C -> B T S C", T=16, S=int(tmp / 16))
        # 获取tensor的维度
        B, T, S, C = tensor.shape
    else:  # weight
        tensor = data
        S, C = tensor.shape
        B = 1
    tensor = tensor.cpu()  # 先分离计算图并移到CPU
    tensor = tensor.numpy()
    # 为每个batch创建和保存独立的图
    for batch_idx in range(B):
        # 创建新的图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.tick_params(axis="z",labelsize=17)
        # 获取当前batch的数据
        if (len(input.shape) == 3):
            data = tensor[batch_idx][0]  # 取第一帧
        else:
            data = tensor
        # 创建坐标网格
        x = np.arange(S)  # 序列位置
        y = np.arange(C)  # 通道维度
        X, Y = np.meshgrid(x, y)
        # 获取Z值（特征值）
        Z = data.T  # 转置以匹配meshgrid的形状
        # 绘制3D表面图
        surf = ax.plot_surface(X, Y, Z,
                               cmap='viridis',
                               edgecolor='none')
        # # 添加颜色条
        # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        # 设置标签
        if (len(input.shape) == 3):
            ax.set_xlabel('Token', fontsize=22)
            ax.set_ylabel('Channel', fontsize=22)
        else:
            ax.set_xlabel('OutputChannel', fontsize=22)
            ax.set_ylabel('InputChannel', fontsize=22)
        # 调整视角
        ax.view_init(elev=30, azim=45)
        # 保存图片
        save_path = os.path.join(save_dir, name)
        plt.savefig(save_path, dpi=600, bbox_inches='tight', transparent=True)
        # 关闭当前图形，释放内存
        plt.close(fig)

    print(f"所有分布图已保存到目录: {save_dir}")

def save_2d_distributions(weight, save_dir='./logs/figure/', name=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnchoredText
    import os
    os.makedirs(save_dir, exist_ok=True)
    if (len(weight.shape) == 3):
        data = weight[0].detach().clone().cpu().numpy()
    else:
        data = weight.detach().clone().cpu().numpy()
    data = data.flatten()
    downsample_step = 200

    # 创建专业级可视化
    plt.figure(figsize=(9, 6))
    plt.tick_params(labelsize=23)
    ax = plt.gca()
    ax.plot(data[::downsample_step],  # 降采样显示
                   linewidth=1,
                   alpha=0.9)

    # 增强可视化元素
    ax.set_ylabel('Value', fontsize=26)

    # 设置现代风格网格线
    ax.grid(True, 
        which='both', 
        linestyle=':', 
        linewidth=0.8, 
        color='#d8dee9', 
        alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path, bbox_inches='tight', dpi=500, transparent=True)
    plt.close()

    print(f"所有分布图已保存到目录: {save_dir}")

class QuantLayer(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear, nn.Conv1d], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, act_quant_mode: str = 'qdiff'):
        super(QuantLayer, self).__init__()
        # self._orginal_module = org_module
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        self.Q1 = None  # rotation
        self.Q2 = None  # rotation
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv1d
        else:
            self.in_features = org_module.in_features
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        # self.org_weight = org_module.weight.data.clone()
        self.org_weight = org_module.weight
        if org_module.bias is not None:
            self.bias = org_module.bias
            # self.org_bias = org_module.bias.data.clone()
            self.org_bias = org_module.bias
        else:
            self.bias = None
            self.org_bias = None
        self.org_module = org_module

        # set use_quant as False, use set_quant_state to set
        self.weight_quant = False
        self.act_quant = False
        self.act_quant_mode = act_quant_mode
        self.disable_act_quant = disable_act_quant

        # initialize quantizer
        if self.weight_quant_params is not None:
            self.weight_quantizer = WeightQuantizer(self.weight_quant_params)
            # self.gptq = GPTQ(self)
            # self.gptq.weight_quantizer = Quantizer_GPTQ()
            # self.gptq.weight_quantizer.configure(
            #     bits=4,
            #     perchannel=True,
            #     sym=True,
            #     mse=False
            # )
        if self.act_quant_params is not None:
            if self.act_quant_params.get('dynamic', False):
                self.act_quantizer = DynamicActQuantizer(self.act_quant_params)
            else:
                self.act_quantizer = ActQuantizer(self.act_quant_params)
        self.split = 0

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.extra_repr = org_module.extra_repr
        # for smooth quant
        smooth_quant_params = act_quant_params.get("smooth_quant", {})
        self.smooth_quant = smooth_quant_params.get("enable", False)
        if self.smooth_quant:
            cur_timerange_id = 0
            self.timerange = smooth_quant_params.get("timerange", [[0, 1000]]) 
            # check the time range
            pre_t = -1
            for r in self.timerange:
                assert r[0] == pre_t + 1 
                pre_t = r[1]
            assert pre_t == 1000

            self.timerange_num = len(self.timerange)  # how many ranges (how many alphas)
            self.act_quantizer.register_buffer("act_scale", None)
            self.channel_wise_scale_type = smooth_quant_params.get("channel_wise_scale_type", "dynamic")
            self.smooth_quant_momentum = smooth_quant_params.get("momentum", 0)
            self.smooth_quant_alpha = smooth_quant_params.get("alpha", None)
            # assert self.timerange_num == len(self.smooth_quant_alpha)
            self.smooth_quant_running_stat = False

        # duquant
        self.init_duquant_params = torch.tensor(0)
        self.flag = 0
        # duquant

    def forward(self, input: torch.Tensor, scale: float = 1.0, split: int = 0, smooth_quant_enable: bool = False):
        # self.cur_timestep_id = glo.get_value('glo_t')[0].to(int)
        # DEBUG_ONLY: test the time of init
        global block
        if split != 0 and self.split != 0:
            assert (split == self.split)
        elif split != 0:
            logger.info(f"split at {split}!")
            self.split = split
            self.set_split()
        if self.smooth_quant:
            # Timestep-wise Quantization
            cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)
            if isinstance(self.smooth_quant_alpha, (list, ListConfig)):
                alpha = self.smooth_quant_alpha[cur_timerange_id]
            else:
                alpha = self.smooth_quant_alpha

            if self.channel_wise_scale_type == "dynamic":
                channel_wise_scale = input.abs().max(dim=-2)[0].pow(alpha).mean(dim=0, keepdim=True) / \
                                     self.weight.abs().max(dim=0)[0].pow(1 - alpha)
            elif "momentum" in self.channel_wise_scale_type:
                if self.smooth_quant_running_stat:
                    cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)  # 为什么取mean：对batchsize取均值
                    if self.act_quantizer.act_scale is None:
                        self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                    if self.act_quantizer.act_scale[cur_timerange_id].abs().mean() == 0:
                        self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                    else:
                        self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[
                                                                             cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (
                                                                                     1 - self.smooth_quant_momentum)
                else:
                    assert self.act_quantizer.act_scale[cur_timerange_id] is not None
                    assert self.act_quantizer.act_scale[cur_timerange_id].mean() != 0
                    if (self.act_quantizer.act_scale[cur_timerange_id] == 0).sum() != 0:
                        zero_mask = self.act_quantizer.act_scale[cur_timerange_id] == 0
                        eps = 1.e-5
                        self.act_quantizer.act_scale[cur_timerange_id][zero_mask] = eps
                        logging.info('act_scale containing zeros, replacing with {}'.format(eps))

                channel_wise_scale = self.act_quantizer.act_scale[cur_timerange_id].pow(alpha) / \
                                     self.weight.abs().max(dim=0)[0].pow(1 - alpha)
            else:
                raise NotImplementedError
            # 画图
            # if block%2 == 0:
            #     save_3d_distributions(input, save_dir=f"./logs/figure/duquant/acl", name=f"block_{(block//2)+1}_mlp_fc1_bf.png")
            # else:
            #     save_3d_distributions(input, save_dir=f"./logs/figure/duquant/acl", name=f"block_{(block//2)+1}_mlp_fc2_bf.png")
            input = input / channel_wise_scale
        else:
            
            # for timeranges, update the act_scale for each timerange respectively
            if not hasattr(self, 'timerange'):
                cur_timerange_id = 0
            else:
                cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)
                print("cur_timerange_id:", cur_timerange_id)
            if getattr(self, "smooth_quant_running_stat", False) and "momentum" in self.channel_wise_scale_type:
                cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                if self.act_quantizer.act_scale is None:
                    self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                if self.act_quantizer.act_scale[cur_timerange_id].abs().mean() == 0:
                    self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                else:
                    # print("######################")
                    self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[
                                                                         cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (
                                                                                 1 - self.smooth_quant_momentum)

        # print(cur_timerange_id) # debug only
        
        if not self.disable_act_quant and self.act_quant:
            if self.split != 0:
                if self.act_quant_mode == 'qdiff':
                    input_0 = self.act_quantizer(input[:, :self.split, :, :])
                    input_1 = self.act_quantizer_0(input[:, self.split:, :, :])
                input = torch.cat([input_0, input_1], dim=1)
            else:
                if self.act_quant_mode == 'qdiff':
                    # duquant
                    
                    
                    if self.flag != cur_timerange_id:
                        self.act_quantizer.init_duquant_params = torch.tensor(0)
                        self.init_duquant_params = torch.tensor(0)
                        self.flag = cur_timerange_id
                    from einops import rearrange
                    tmp, bth = input.shape[1] // 16, input.shape[0]
                    # input = rearrange(input, "B (T S) C -> B T S C", T=16, S=tmp)
                    # 取中间的帧为基准生成旋转矩阵等，其他帧进行共享
                    # for i in range(bth):
                    #     input[i][7] = self.act_quantizer.init_duquant(input[i][7], cur_timerange_id)
                    #     for frames in range(16):  # 16帧
                    #         if frames != 7:
                    #             input[i][frames] = self.act_quantizer.init_duquant(input[i][frames], cur_timerange_id)
                    # print(cur_timerange_id)
                    # tmp = input.detach().clone()
                    # print("****")
                    # print(cur_timerange_id)
                    for i in range(bth):
                        input[i] = self.act_quantizer.init_duquant(input[i], cur_timerange_id)
                  
                    # # duquant画图
                    # if block%2 == 0:
                    #     save_3d_distributions(input, save_dir=f"./logs/figure/duquant/acl", name=f"block_{(block//2)+1}_mlp_fc1_r.png")
                    #     # save_2d_distributions(tmp, save_dir=f"./logs/figure/duquant/ac2D", name=f"block_{(block//2)+1}_mlp_fc1_w.png")
                    # else:
                    #     save_3d_distributions(input, save_dir=f"./logs/figure/duquant/acl", name=f"block_{(block//2)+1}_mlp_fc2_r,png")
                    #     # save_2d_distributions(tmp, save_dir=f"./logs/figure/duquant/ac2D", name=f"block_{(block//2)+1}_mlp_fc2_w.png")
                    # # duquant
                    input = self.act_quantizer(input)

        if self.weight_quant:
            # print("时间：", glo.get_value('glo_t'))
            if self.split != 0:
                weight_0 = self.weight_quantizer(self.weight[:, :self.split, ...])
                weight_1 = self.weight_quantizer_0(self.weight[:, self.split:, ...])
                weight = torch.cat([weight_0, weight_1], dim=1)
            else:
                if self.smooth_quant:
                    # during the weight init stage
                    if self.weight_quantizer.timestep_wise is None:  # reinit the weight_quantizer
                        self.weight_quantizer.timestep_wise = True
                        self.weight_quantizer.n_timestep = len(self.timerange)
                        if not self.weight_quantizer.init_done:
                            self.weight_quantizer.delta_list = None  # reset as none for automatic init of delta_list during forward
                            self.weight_quantizer.zero_point_list = None  # reset as none for automatic init of delta_list during forward
                    self.weight_quantizer.cur_timestep_id = cur_timerange_id
                    # weight = self.weight_quantizer(self.weight * channel_wise_scale)  # duquant

                    # # duquant
                    # if block%2 == 0:
                    #     save_2d_distributions(self.weight * channel_wise_scale, save_dir=f"./logs/figure/duquant/weightT/", name=f"block_{(block//2)+1}_mlp_fc1_bf.png")
                    # else:
                    #     save_2d_distributions(self.weight * channel_wise_scale, save_dir=f"./logs/figure/duquant/weightT/", name=f"block_{(block//2)+1}_mlp_fc2_bf,png")
                    if not self.disable_act_quant and self.act_quant and self.act_quant_mode == 'qdiff':    
                        if not self.init_duquant_params and self.weight_quantizer.init_done == False:
                            self.weight_quantizer.copy_duquant_params(self.act_quantizer)
                            self.init_duquant_params = torch.tensor(1)
                        weight = self.weight_quantizer.init_duquant(self.weight * channel_wise_scale, cur_timerange_id)
                        weight = self.weight_quantizer(weight)
                        
                    else:
                        weight = self.weight_quantizer(self.weight * channel_wise_scale)
                    
                    # if block%2 == 0:
                    #     save_2d_distributions(weight, save_dir=f"./logs/figure/duquant/weightT/", name=f"block_{(block//2)+1}_mlp_fc1_ap.png")
                    # else:
                    #     save_2d_distributions(weight, save_dir=f"./logs/figure/duquant/weightT/", name=f"block_{(block//2)+1}_mlp_fc2_ap,png")
                    block += 1
                    # weight = self.weight_quantizer(self.weight * channel_wise_scale)
                    torch.cuda.empty_cache()  # 清理
                    # duquant
                else:
                    if not self.disable_act_quant and self.act_quant and self.act_quant_mode == 'qdiff':
                        if not self.init_duquant_params and self.weight_quantizer.init_done == False:
                            self.weight_quantizer.copy_duquant_params(self.act_quantizer)
                            self.init_duquant_params = torch.tensor(1)
                        weight = self.weight_quantizer.init_duquant(self.weight, cur_timerange_id)
                        weight = self.weight_quantizer(weight)
                    else:
                        weight = self.weight_quantizer(self.weight)
                    # weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            if self.smooth_quant:
                weight = self.org_weight * channel_wise_scale
            else:
                weight = self.org_weight
            bias = self.org_bias

        # if self.smooth_quant:
        #     import ipdb; ipdb.set_trace()
        # t_quantizer_init_done = time.time()
        # logging.info('quantizer init elapsed time:{}'.format(t_quantizer_init_done - t_start))

        # if(type(self.fwd_func)==F.linear):
        #     print(input.shape, weight.shape)

        if weight.dtype == torch.float32 and input.dtype == torch.float16:
            weight = weight.to(torch.float16)

        # DEBUG_ONLY: print the dtype
        # if bias == None:
        # print(input.dtype, weight.dtype)
        # else:
        # print(input.dtype, weight.dtype, bias.dtype)
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)  # 在输出的channel上进行channel_wise的量化
        out = self.activation_function(out)

        if torch.isnan(out).any():
            logging.info('nan exist in the activation')
            import ipdb;
            ipdb.set_trace()

        # DEBUG_ONLY
        # if self.smooth_quant:
        # out_golden = self.fwd_func(input*channel_wise_scale, weight/channel_wise_scale, bias, **self.fwd_kwargs)
        # print('{}, error w.o, smooth quant'.format(self.act_quantizer.module_name), (out - out_golden).abs().mean())

        # torch.cuda.empty_cache()  # DEBUG: memory accumulate

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):  # 判断是否设置为量化模式！！！
        self.weight_quant = weight_quant
        self.act_quant = act_quant

    def get_quant_state(self):
        return self.weight_quant, self.act_quant

    def set_split(self):
        self.weight_quantizer_0 = WeightQuantizer(self.weight_quant_params)
        if self.act_quant_mode == 'qdiff':
            self.act_quantizer_0 = ActQuantizer(self.act_quant_params)

    # def set_running_stat(self, running_stat: bool):
    # if self.act_quant_mode == 'qdiff':
    # self.act_quantizer.running_stat = running_stat
    # if self.split != 0:
    # self.act_quantizer_0.running_stat = running_stat

    # def __getattr__(self, name):
    #     try:
    #         return super().__getattr__(name)
    #     except AttributeError:
    #         return getattr(self._orginal_module, name) 

    # def __getattr__(self, name: str) -> Union[torch.Tensor, 'Module']:
    #     return self._orginal_module.__getattr__(name)


class test:
    def p(self):
        print(123)
