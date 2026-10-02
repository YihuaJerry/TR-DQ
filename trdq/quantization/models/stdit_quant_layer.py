import torch
from trdq.quantization.quantizer.base_quantizer import WeightQuantizer, ActQuantizer, StraightThrough
from trdq.quantization.models.quant_layer import QuantLayer, find_interval
from omegaconf import ListConfig

'''
Utility QuantLayers for STDiT temporal/spatial attn layer linears
'''

import os
import numpy as np
import matplotlib.pyplot as plt


def save_3d_distributions(input, save_dir='./distribution_plots/'):
    """
    为tensor中的每个batch创建并保存独立的3D分布图

    参数:
    tensor: 形状为[BS, T*S, C]的numpy数组
    save_dir: 保存图片的目录路径
    """
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

        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        # 设置标签
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Channel')
        ax.set_zlabel('Value')
        ax.set_title(f'Distribution for Batch {batch_idx + 1}')

        # 调整视角
        ax.view_init(elev=30, azim=45)

        # 保存图片
        save_path = os.path.join(save_dir, f'distribution_batch_{batch_idx + 1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # 关闭当前图形，释放内存
        plt.close(fig)

    print(f"所有分布图已保存到目录: {save_dir}")


class QuantSpatialAttnLinear(QuantLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor, scale: float = 1.0, split: int = 0):
        # check the n_spatial/temporal_token num in act_quant_config is True
        BS = input.shape[0]//self.act_quant_params['n_temporal_token']
        T = self.act_quant_params['n_temporal_token']
        S = self.act_quant_params['n_spatial_token']
        C = input.shape[2]
        assert input.shape[1] == S

        if self.smooth_quant:
            cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)
            if isinstance(self.smooth_quant_alpha, (list, ListConfig)):
                alpha = self.smooth_quant_alpha[cur_timerange_id]
            else:
                alpha = self.smooth_quant_alpha

            if self.channel_wise_scale_type == "dynamic":
                channel_wise_scale = input.abs().max(dim=-2)[0].pow(alpha).mean(dim=0, keepdim=True) / self.weight.abs().max(dim=0)[0].pow(1 - alpha)
            elif "momentum" in self.channel_wise_scale_type:
                if self.smooth_quant_running_stat:
                    cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                    if self.act_quantizer.act_scale is None:
                        self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                    if self.act_quantizer.act_scale[cur_timerange_id].abs().mean()==0:
                        self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                    else:
                        self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)
                else:
                    assert self.act_quantizer.act_scale[cur_timerange_id] is not None
                    assert self.act_quantizer.act_scale[cur_timerange_id].mean() != 0
                    if (self.act_quantizer.act_scale[cur_timerange_id] == 0).sum() != 0:
                        zero_mask = self.act_quantizer.act_scale[cur_timerange_id] == 0
                        eps = 1.e-5
                        self.act_quantizer.act_scale[cur_timerange_id][zero_mask] = eps
                        logging.info('act_scale containing zeros, replacing with {}'.format(eps))

                channel_wise_scale = self.act_quantizer.act_scale[cur_timerange_id].pow(alpha) / self.weight.abs().max(dim=0)[0].pow(1 - alpha)
            else:
                raise NotImplementedError
            input = input / channel_wise_scale
        else:
            if not hasattr(self, 'timerange'):
                cur_timerange_id = 0
            else:
                cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)
            if getattr(self, "smooth_quant_running_stat", False) and "momentum" in self.channel_wise_scale_type:
                cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                if self.act_quantizer.act_scale is None:
                    self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                if self.act_quantizer.act_scale[cur_timerange_id].abs().mean()==0:
                    self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                else:
                    self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)

        if not self.disable_act_quant and self.act_quant:
            # 条件和非条件路径的相似性分析
            # convert the dim into [bs, n_token, c]
            input = input.reshape([BS,T*S,C])
            # duquant
            # save_3d_distributions(input, debug_output_dir)
            if self.flag != cur_timerange_id:
                self.act_quantizer.init_duquant_params = torch.tensor(0)
                self.init_duquant_params = torch.tensor(0)
                self.flag = cur_timerange_id
            from einops import rearrange
            tmp, bth = input.shape[1] // 16, input.shape[0]
            for i in range(bth):
                input[i] = self.act_quantizer.init_duquant(input[i], cur_timerange_id)
            # input = rearrange(input, "B T S C -> B (T S) C", T=16, S=tmp)
            # duquant
            # save_3d_distributions(input, debug_output_dir)
            input = self.act_quantizer(input)
            # convert back
            input = input.reshape([BS*T,S,C])

        if self.weight_quant:
            if self.smooth_quant:
                # during the weight init stage
                if self.weight_quantizer.timestep_wise is None: # reinit the weight_quantizer
                    self.weight_quantizer.timestep_wise = True
                    self.weight_quantizer.n_timestep = len(self.timerange)
                    if not self.weight_quantizer.init_done:
                        self.weight_quantizer.delta_list = None  # reset as none for aautomatic init of delta_list during forward
                        self.weight_quantizer.zero_point_list = None  # reset as none for aautomatic init of delta_list during forward
                self.weight_quantizer.cur_timestep_id = cur_timerange_id
                # weight = self.weight_quantizer(self.weight * channel_wise_scale)  # duquant
                # duquant
                if not self.init_duquant_params and self.weight_quantizer.init_done == False:
                    self.weight_quantizer.copy_duquant_params(self.act_quantizer)
                    self.init_duquant_params = torch.tensor(1)

                # weight = self.weight_quantizer.init_duquant(self.weight * channel_wise_scale, cur_timerange_id)
                # save_3d_distributions(self.weight * channel_wise_scale, debug_output_dir)
                weight = self.weight_quantizer.init_duquant(self.weight * channel_wise_scale, cur_timerange_id)
                # save_3d_distributions(weight, debug_output_dir)
                weight = self.weight_quantizer(weight)
                # duquant
            else:
                weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias

        if weight.dtype == torch.float32 and input.dtype == torch.float16:
            weight = weight.to(torch.float16)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)

        return out

class QuantTemporalAttnLinear(QuantLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor, scale: float = 1.0, split: int = 0):
        # check the n_spatial/temporal_token num in act_quant_config is True
        BS = input.shape[0]//self.act_quant_params['n_spatial_token']
        T = self.act_quant_params['n_temporal_token']
        S = self.act_quant_params['n_spatial_token']
        C = input.shape[2]
        assert input.shape[1] == T

        if self.smooth_quant:
            cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)
            if isinstance(self.smooth_quant_alpha, (list, ListConfig)):
                alpha = self.smooth_quant_alpha[cur_timerange_id]
            else:
                alpha = self.smooth_quant_alpha

            if self.channel_wise_scale_type == "dynamic":
                channel_wise_scale = input.abs().max(dim=-2)[0].pow(alpha).mean(dim=0, keepdim=True) / self.weight.abs().max(dim=0)[0].pow(1 - alpha)
            elif "momentum" in self.channel_wise_scale_type:
                if self.smooth_quant_running_stat:
                    cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                    if self.act_quantizer.act_scale is None:
                        self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                    if self.act_quantizer.act_scale[cur_timerange_id].abs().mean()==0:
                        self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                    else:
                        self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)
                else:
                    assert self.act_quantizer.act_scale[cur_timerange_id] is not None
                    assert self.act_quantizer.act_scale[cur_timerange_id].mean() != 0
                    if (self.act_quantizer.act_scale[cur_timerange_id] == 0).sum() != 0:
                        zero_mask = self.act_quantizer.act_scale[cur_timerange_id] == 0
                        eps = 1.e-5
                        self.act_quantizer.act_scale[cur_timerange_id][zero_mask] = eps
                        logging.info('act_scale containing zeros, replacing with {}'.format(eps))

                channel_wise_scale = self.act_quantizer.act_scale[cur_timerange_id].pow(alpha) / self.weight.abs().max(dim=0)[0].pow(1 - alpha)
            else:
                raise NotImplementedError
            input = input / channel_wise_scale
        else:
            if not hasattr(self, 'timerange'):
                cur_timerange_id = 0
            else:
                cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)
            if getattr(self, "smooth_quant_running_stat", False) and "momentum" in self.channel_wise_scale_type:
                cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                if self.act_quantizer.act_scale is None:
                    self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                if self.act_quantizer.act_scale[cur_timerange_id].abs().mean()==0:
                    self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                else:
                    self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)

        if not self.disable_act_quant and self.act_quant:
            # convert the dim into [bs, n_token, c]
            input = input.reshape([BS,S*T,C])
            # duquant
            if self.flag != cur_timerange_id:
                self.act_quantizer.init_duquant_params = torch.tensor(0)
                self.init_duquant_params = torch.tensor(0)
                self.flag = cur_timerange_id
            from einops import rearrange
            tmp, bth = input.shape[1] // 16, input.shape[0]
            for i in range(bth):
                input[i] = self.act_quantizer.init_duquant(input[i], cur_timerange_id)
            # duquant
            input = self.act_quantizer(input)
            # convert back
            input = input.reshape([BS*S,T,C])

        if self.weight_quant:
            if self.smooth_quant:
                # during the weight init stage
                if self.weight_quantizer.timestep_wise is None: # reinit the weight_quantizer
                    self.weight_quantizer.timestep_wise = True
                    self.weight_quantizer.n_timestep = len(self.timerange)
                    if not self.weight_quantizer.init_done:
                        self.weight_quantizer.delta_list = None  # reset as none for aautomatic init of delta_list during forward
                        self.weight_quantizer.zero_point_list = None  # reset as none for aautomatic init of delta_list during forward
                self.weight_quantizer.cur_timestep_id = cur_timerange_id
                # weight = self.weight_quantizer(self.weight * channel_wise_scale)  # duquant
                # duquant
                if not self.init_duquant_params and self.weight_quantizer.init_done == False:
                    self.weight_quantizer.copy_duquant_params(self.act_quantizer)
                    self.init_duquant_params = torch.tensor(1)

                # weight = self.weight_quantizer.init_duquant(self.weight * channel_wise_scale, cur_timerange_id)
                weight = self.weight_quantizer.init_duquant(self.weight * channel_wise_scale, cur_timerange_id)
                weight = self.weight_quantizer(weight)
                # duquant
            else:
                weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias

        if weight.dtype == torch.float32 and input.dtype == torch.float16:
            weight = weight.to(torch.float16)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)

        return out

class QuantCrossAttnLinear(QuantLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # TODO: new forward, cleaner
    def forward(self, input: torch.Tensor, scale: float = 1.0, split: int = 0):
        # Need to handle both Q & KV
        # Q_Linear: [BS, T*S, C]
        # KV_Linear: [1, BS*n_prompt, C]

        T = self.act_quant_params['n_temporal_token']
        S = self.act_quant_params['n_spatial_token']
        C = input.shape[2]

        if input.shape[1] == T*S:
            layer_type = "q"
            BS = input.shape[0]
        elif input.shape[0] == 1:
            layer_type = "kv"
            BS = input.shape[1]//self.act_quant_params['n_prompt']
            n_prompt = self.act_quant_params['n_prompt']
        else:
            print('illegeal shape.')
            # import ipdb; ipdb.set_trace()

        if self.smooth_quant:
            cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)

            if isinstance(self.smooth_quant_alpha, (list, ListConfig)):
                alpha = self.smooth_quant_alpha[cur_timerange_id]
            else:
                alpha = self.smooth_quant_alpha

            if self.channel_wise_scale_type == "dynamic":
                channel_wise_scale = input.abs().max(dim=-2)[0].pow(alpha).mean(dim=0, keepdim=True) / self.weight.abs().max(dim=0)[0].pow(1 - alpha)
            elif "momentum" in self.channel_wise_scale_type:
                if self.smooth_quant_running_stat:
                    cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                    if self.act_quantizer.act_scale is None:
                        self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                    if self.act_quantizer.act_scale[cur_timerange_id].abs().mean()==0:
                        self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                    else:
                        self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)
                else:
                    assert self.act_quantizer.act_scale[cur_timerange_id] is not None
                    assert self.act_quantizer.act_scale[cur_timerange_id].mean() != 0
                    if (self.act_quantizer.act_scale[cur_timerange_id] == 0).sum() != 0:
                        zero_mask = self.act_quantizer.act_scale[cur_timerange_id] == 0
                        eps = 1.e-5
                        self.act_quantizer.act_scale[cur_timerange_id][zero_mask] = eps
                        logging.info('act_scale containing zeros, replacing with {}'.format(eps))

                channel_wise_scale = self.act_quantizer.act_scale[cur_timerange_id].pow(alpha) / self.weight.abs().max(dim=0)[0].pow(1 - alpha)

            else:
                raise NotImplementedError
            input = input / channel_wise_scale
        else:
            if not hasattr(self, 'timerange'):
                cur_timerange_id = 0
            else:
                cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)
            if getattr(self, "smooth_quant_running_stat", False) and "momentum" in self.channel_wise_scale_type:
                cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                if self.act_quantizer.act_scale is None:
                    self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                if self.act_quantizer.act_scale[cur_timerange_id].abs().mean()==0:
                    self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                else:
                    self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)

        if not self.disable_act_quant and self.act_quant:
            # convert the dim into [bs, n_token, c]
            if layer_type == 'q':
                # duquant
                if self.flag != cur_timerange_id:
                    self.act_quantizer.init_duquant_params = torch.tensor(0)
                    self.init_duquant_params = torch.tensor(0)
                    self.flag = cur_timerange_id
                tmp, bth = input.shape[1] // 16, input.shape[0]
                for i in range(bth):
                    input[i] = self.act_quantizer.init_duquant(input[i], cur_timerange_id)
                # duquant
                input = self.act_quantizer(input)
            elif layer_type == 'kv':
                # INFO: when mask_select=True
                # it only supports dynamic quant
                if not self.act_quant_params.get('dynamic',False):
                    if self.act_quant_params.per_group is False:  # no need to reshape for tensor-wise quant
                        # duquant
                        if self.flag != cur_timerange_id:
                            self.act_quantizer.init_duquant_params = torch.tensor(0)
                            self.init_duquant_params = torch.tensor(0)
                            self.flag = cur_timerange_id
                        tmp, bth = input.shape[1] // 16, input.shape[0]
                        for i in range(bth):
                            input[i] = self.act_quantizer.init_duquant(input[i], cur_timerange_id)
                        # duquant
                        input = self.act_quantizer(input)
                    else:
                        input = input.reshape([BS,n_prompt,C])
                        # duquant
                        if self.flag != cur_timerange_id:
                            self.act_quantizer.init_duquant_params = torch.tensor(0)
                            self.init_duquant_params = torch.tensor(0)
                            self.flag = cur_timerange_id
                        tmp, bth = input.shape[1] // 16, input.shape[0]
                        for i in range(bth):
                            input[i] = self.act_quantizer.init_duquant(input[i], cur_timerange_id)
                        # duquant
                        input = self.act_quantizer(input)
                        input = input.reshape([1,BS*n_prompt,C])
                else:
                    # directly assign N_batch*prompt quant_params for each token
                    # duquant
                    if self.flag != cur_timerange_id:
                        self.act_quantizer.init_duquant_params = torch.tensor(0)
                        self.init_duquant_params = torch.tensor(0)
                        self.flag = cur_timerange_id
                    tmp, bth = input.shape[1] // 16, input.shape[0]
                    for i in range(bth):
                        input[i] = self.act_quantizer.init_duquant(input[i], cur_timerange_id)
                    # duquant
                    input = self.act_quantizer(input)

        if self.weight_quant:
            if self.smooth_quant:
                if not self.weight_quantizer.init_done:
                    if self.weight_quantizer.timestep_wise is None: # reinit the weight_quantizer
                        self.weight_quantizer.timestep_wise = True
                        self.weight_quantizer.n_timestep = len(self.timerange)
                        if not self.weight_quantizer.init_done:
                            self.weight_quantizer.delta_list = None  # reset as none for aautomatic init of delta_list during forward
                            self.weight_quantizer.zero_point_list = None  # reset as none for aautomatic init of delta_list during forward
                self.weight_quantizer.cur_timestep_id = cur_timerange_id
                # weight = self.weight_quantizer(self.weight * channel_wise_scale)  # duquant
                # duquant
                if not self.init_duquant_params and self.weight_quantizer.init_done == False:
                    self.weight_quantizer.copy_duquant_params(self.act_quantizer)
                    self.init_duquant_params = torch.tensor(1)

                # weight = self.weight_quantizer.init_duquant(self.weight * channel_wise_scale, cur_timerange_id)
                weight = self.weight_quantizer.init_duquant(self.weight * channel_wise_scale, cur_timerange_id)
                weight = self.weight_quantizer(weight)
                # duquant
            else:
                weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias

        if weight.dtype == torch.float32 and input.dtype == torch.float16:
            weight = weight.to(torch.float16)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)

        return out


