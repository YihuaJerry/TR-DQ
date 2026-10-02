from functools import partial

from torch import nn as nn
from itertools import repeat
import collections.abc
from einops import rearrange
import os
import numpy as np
import matplotlib.pyplot as plt


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

flag = 0  # 层数

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

# def normalize(x):
#     # 使用z-score标准化（去均值，除以标准差）
#     mean = np.mean(x)
#     std = np.std(x)
#     return (x - mean) / std if std != 0 else x  # 防止标准差为零的情况
#
def cosine_similarity(x1, x2):  # 计算余弦相似度
    dot_product = np.sum(x1 * x2, axis=-1)
    norm_x1 = np.linalg.norm(x1, axis=-1)
    norm_x2 = np.linalg.norm(x2, axis=-1)
    return dot_product / (norm_x1 * norm_x2)

def frames(x, save_path):
    data = x.detach().clone()
    tmp = data.shape[1]
    data = rearrange(data, "B (T S) C -> B T S C", T=16, S=int(tmp / 16))
    x1 = data[0][0].to('cpu').numpy()
    x2 = data[0][1].to('cpu').numpy()
    similarity_matrix = cosine_similarity(x1, x2)

    # 使用 seaborn 绘制相关性热力图
    plt.figure(figsize=(20, 16))
    plt.plot(similarity_matrix, label='Data Values', color='b')
    plt.axhline(y=0.9, color='r', linestyle='--', label='Threshold 0.95')

    # 添加标题和标签
    plt.title("Frames Similarity")
    plt.xlabel("Channel")
    plt.ylabel("Similarity")
    plt.legend()
    if not os.path.exists(save_path):  # 如果路径不存在
        os.makedirs(save_path)
    save_path = save_path + f"blocks_{flag % 28}.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')  # dpi、边界可按需调整
    plt.close()

def figure_save(x, flag, save_path, tmp):
    data = x.detach().clone()
    tmp = data.shape[1]
    data = rearrange(data, "B (T S) C -> B T S C", T=16, S=int(tmp / 16))
    data = data[0][0].to('cpu')

    num_x = int(tmp / 16)
    num_y = 1152

    step_x = data.shape[0] // num_x  # 在行方向的步长
    step_y = data.shape[1] // num_y  # 在列方向的步长

    # 下采样：只取部分行、列
    data_sub = data[::step_x, ::step_y]  # shape -> (num_x, num_y)

    # 现在 data_sub.shape 大约为 (128, 128)，可视化更容易
    data_sub = data_sub.numpy()  # 转为 numpy 方便 matplotlib 处理

    x_coords = np.arange(data_sub.shape[0])  # [0, 1, ..., 127]
    y_coords = np.arange(data_sub.shape[1])  # [0, 1, ..., 127]

    # 生成网格
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

    fig = plt.figure(figsize=(30, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        X,
        Y,
        data_sub,
        cmap='viridis',  # 颜色映射，可选
        linewidth=0,  # 网格线宽度
        antialiased=True,  # 抗锯齿
    )

    # 可以加一个颜色条
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_title("3D Surface of data_sub")
    ax.set_xlabel("InputC")
    ax.set_ylabel("OutputC")
    ax.set_zlabel("Values")


    if not os.path.exists(save_path):  # 如果路径不存在
        os.makedirs(save_path)

    save_path = save_path + f"blocks_{flag%28}.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')  # dpi、边界可按需调整
    plt.close(fig)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x, set_ipdb=False):
        global flag
        # debug
        steps = [5, 10, 15, 20]
        # if flag <=28*20-1:
        #     save_path_ = f"./logs/figure/uncond/frames_0/steps_{(flag//28)+1}"
        # else:
        #     save_path_ = f"./logs/figure/cond/frames_0/steps_{(flag//28+1)}"
        # if (flag//28)+1 in steps:
        #     save_path = save_path_ + "/fc1/"
        #     figure_save(x, flag, save_path, 1)
        # if (flag//28)+1 in steps:
        #     save_path = save_path_ + "/fc1/"
        #     figure_save(x, flag, save_path, 1)
        # if (flag//28)+1 in steps and (flag % 28 == 27 or flag % 28 == 0):  # blocks[27/0]才进去
        #     save_path = save_path_ + "/fc1/"
        #     frames(x, save_path)
        x = self.fc1(x)
        if set_ipdb:
            import ipdb; ipdb.set_trace()
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        if set_ipdb:
            import ipdb; ipdb.set_trace()
        # if (flag // 28) + 1 in steps:
        #     save_path = save_path_ + "/fc2/"
        #     figure_save(x, flag, save_path, 2)
        # if (flag//28)+1 in steps and (flag % 28 == 27 or flag % 28 == 0):  # blocks[27/0]才进去
        #     save_path = save_path_ + "/fc2/"
        #     frames(x, save_path)
        # flag += 1
        x = self.fc2(x)
        if set_ipdb:
            import ipdb; ipdb.set_trace()
        x = self.drop2(x)
        return x
