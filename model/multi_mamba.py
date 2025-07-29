# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from einops import rearrange, repeat
import logging

try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn_no_out_proj
except ImportError:
    mamba_inner_fn_no_out_proj = None


class MultiScan(nn.Module):
    ALL_CHOICES = ('h', 'h_flip', 'v', 'v_flip', 'c2', 'c5')

    def __init__(self, dim, choices=None, token_size=(14, 14), win_size=8):
        super().__init__()
        self.token_size = token_size
        self.win_size = win_size
        if choices is None:
            self.choices = MultiScan.ALL_CHOICES
            self.norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False) for _ in self.choices])
            self.weights = nn.Parameter(1e-3 * torch.randn(len(self.choices), 1, 1, 1))
            self._iter = 0
            self.logger = logging.getLogger()
            self.search = True
        else:
            self.choices = choices
            self.search = False

        # XQY
        self.local_cluster_F = GNNLocalCluster0_F(dim=dim, w_size=self.win_size, k_neighbors=9)
        # self.local_cluster_F = EnhancedGNNLocalCluster(dim=dim, w_size=self.win_size, k_neighbors=16)

    def forward(self, xs):
        """
        Input @xs: [[B, L, D], ...]
        """
        if self.search:
            weights = self.weights.softmax(0)
            xs = [norm(x) for norm, x in zip(self.norms, xs)]
            xs = torch.stack(xs) * weights
            x = xs.sum(0)
            if self._iter % 200 == 0:
                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    self.logger.info(str(weights.detach().view(-1).tolist()))
            self._iter += 1
        else:
            x = torch.stack(xs).sum(0)
            # print("x2")
        return x

    def multi_scan(self, x):
        """
        Input @x: shape [B, L, D]
        """
        xs = []
        x = self.local_cluster_F(x)
        for direction in self.choices:
            xs.append(self.scan(x, direction))
        # print(self.choices)
        return xs

    def multi_reverse(self, xs):
        new_xs = []
        for x, direction in zip(xs, self.choices):
            new_xs.append(self.reverse(x, direction))
        return new_xs

    def scan(self, x, direction='h'):
        """
        Input @x: shape [B, L, D] or [B, C, H, W]
        Return torch.Tensor: shape [B, D, L]
        """
        H, W = self.token_size
        if len(x.shape) == 3:
            if direction == 'h':
                return x.transpose(-2, -1)
            elif direction == 'h_flip':
                return x.transpose(-2, -1).flip([-1])
            elif direction == 'v':
                return rearrange(x, 'b (h w) d -> b d (w h)', h=H, w=W)
            elif direction == 'v_flip':
                return rearrange(x, 'b (h w) d -> b d (w h)', h=H, w=W).flip([-1])
            elif direction.startswith('c'):
                # K = int(direction[1:].split('_')[0])
                # flip = direction.endswith('flip')
                # return LocalScanTriton.apply(x.transpose(-2, -1), K, flip, H, W)
                return self.local_cluster(x)
            else:
                raise RuntimeError(f'Direction {direction} not found.')
        elif len(x.shape) == 4:
            if direction == 'h':
                return x.flatten(2)
            elif direction == 'h_flip':
                return x.flatten(2).flip([-1])
            elif direction == 'v':
                return rearrange(x, 'b d h w -> b d (w h)', h=H, w=W)
            elif direction == 'v_flip':
                return rearrange(x, 'b d h w -> b d (w h)', h=H, w=W).flip([-1])
            elif direction.startswith('c'):
                # K = int(direction[1:].split('_')[0])
                # flip = direction.endswith('flip')
                # return LocalScanTriton.apply(x, K, flip, H, W).flatten(2)
                return self.local_cluster(x)
            # elif direction.startswith('f'):
            #     # 先进行聚集操作
            #     # out = self.local_cluster_F(x)
            #     # 根据方向进行 flatten/rearrange
            #     if direction == 'f_h_9':
            #         return out.flatten(2)
            #     elif direction == 'f_h_flip_9':
            #         return out.flatten(2).flip([-1])
            #     elif direction == 'f_v_9':
            #         return rearrange(out, 'b d h w -> b d (w h)', h=H, w=W)
            #     elif direction == 'f_v_flip_9':
            #         return rearrange(out, 'b d h w -> b d (w h)', h=H, w=W).flip([-1])
            else:
                raise RuntimeError(f'Direction {direction} not found.')

    def reverse(self, x, direction='h'):
        """
        Input @x: shape [B, D, L]
        Return torch.Tensor: shape [B, D, L]
        """
        H, W = self.token_size
        if direction == 'h':
            return x
        elif direction == 'h_flip':
            return x.flip([-1])
        elif direction == 'v':
            return rearrange(x, 'b d (h w) -> b d (w h)', h=H, w=W)
        elif direction == 'v_flip':
            return rearrange(x.flip([-1]), 'b d (h w) -> b d (w h)', h=H, w=W)
        elif direction.startswith('c'):
            return x
        elif direction == 'f_h_9':
            return x
        elif direction == 'f_h_flip_9':
            return x.flip([-1])
        elif direction == 'f_v_9':
            return rearrange(x, 'b d (h w) -> b d (w h)', h=H, w=W)
        elif direction == 'f_v_flip_9':
            return rearrange(x.flip([-1]), 'b d (h w) -> b d (w h)', h=H, w=W)
        else:
            raise RuntimeError(f'Direction {direction} not found.')

    def __repr__(self):
        scans = ', '.join(self.choices)
        return super().__repr__().replace(self.__class__.__name__, f'{self.__class__.__name__}[{scans}]')


class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        # self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        # self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        # x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]

        attn = c_attn  # * s_attn  # [B, N, C]
        return ori_x * attn


class MultiMamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=None,
            device=None,
            dtype=None,
            bimamba_type="none",
            directions=None,
            token_size=(14, 14),
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.token_size = token_size

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.activation = "silu"
        self.act = nn.SiLU()

        self.multi_scan = MultiScan(self.d_inner, choices=directions, token_size=token_size)
        '''new for search'''
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        for i in range(len(self.multi_scan.choices)):
            setattr(self, f'A_log_{i}', nn.Parameter(A_log))
            getattr(self, f'A_log_{i}')._no_weight_decay = True

            conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            setattr(self, f'conv1d_{i}', conv1d)

            x_proj = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            setattr(self, f'x_proj_{i}', x_proj)

            dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = self.dt_rank ** -0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

            setattr(self, f'dt_proj_{i}', dt_proj)

            D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            D._no_weight_decay = True
            setattr(self, f'D_{i}', D)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.attn = BiAttn(self.d_inner)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        xz = self.in_proj(hidden_states)

        xs = self.multi_scan.multi_scan(xz)  # [[BDL], [BDL], ...]
        outs = []
        for i, xz in enumerate(xs):
            # xz = rearrange(xz, "b l d -> b d l")
            A = -torch.exp(getattr(self, f'A_log_{i}').float())
            conv1d = getattr(self, f'conv1d_{i}')
            x_proj = getattr(self, f'x_proj_{i}')
            dt_proj = getattr(self, f'dt_proj_{i}')
            D = getattr(self, f'D_{i}')

            out = mamba_inner_fn_no_out_proj(
                xz,
                conv1d.weight,
                conv1d.bias,
                x_proj.weight,
                dt_proj.weight,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                D,
                delta_bias=dt_proj.bias.float(),
                delta_softplus=True,
            )
            outs.append(out)

        outs = self.multi_scan.multi_reverse(outs)
        outs = [self.attn(rearrange(out, 'b d l -> b l d')) for out in outs]
        out = self.multi_scan(outs)
        out = F.linear(out, self.out_proj.weight, self.out_proj.bias)

        return out


try:
    import selective_scan_cuda_oflex
except:
    selective_scan_cuda_oflex = None


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None


class MultiVMamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=None,
            device=None,
            dtype=None,
            bimamba_type="none",
            directions=None,
            token_size=(14, 14),
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.token_size = token_size

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.activation = "silu"
        self.act = nn.SiLU()

        self.multi_scan = MultiScan(self.d_inner, choices=directions, token_size=token_size)
        '''new for search'''
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        for i in range(len(self.multi_scan.choices)):
            setattr(self, f'A_log_{i}', nn.Parameter(A_log))
            getattr(self, f'A_log_{i}')._no_weight_decay = True

            x_proj = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            setattr(self, f'x_proj_{i}', x_proj)

            conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            setattr(self, f'conv1d_{i}', conv1d)

            dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = self.dt_rank ** -0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

            setattr(self, f'dt_proj_{i}', dt_proj)

            D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            D._no_weight_decay = True
            setattr(self, f'D_{i}', D)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.attn = BiAttn(self.d_inner)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch_size, seq_len, dim = hidden_states.shape
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=2)
        z = self.act(z)

        xs = self.multi_scan.multi_scan(x)
        outs = []
        for i, xz in enumerate(xs):
            xz = rearrange(xz, "b l d -> b d l")
            A = -torch.exp(getattr(self, f'A_log_{i}').float())
            x_proj = getattr(self, f'x_proj_{i}')
            conv1d = getattr(self, f'conv1d_{i}')
            dt_proj = getattr(self, f'dt_proj_{i}')
            D = getattr(self, f'D_{i}')

            xz = conv1d(xz)[:, :, :seq_len]
            xz = self.act(xz)

            N = A.shape[-1]
            R = dt_proj.weight.shape[-1]

            x_dbl = F.linear(rearrange(xz, 'b d l -> b l d'), x_proj.weight)
            dts, B, C = torch.split(x_dbl, [R, N, N], dim=2)
            dts = F.linear(dts, dt_proj.weight)

            dts = rearrange(dts, 'b l d -> b d l')
            B = rearrange(B, 'b l d -> b 1 d l')
            C = rearrange(C, 'b l d -> b 1 d l')
            D = D.float()
            delta_bias = dt_proj.bias.float()

            out = SelectiveScanOflex.apply(xz.contiguous(), dts.contiguous(), A.contiguous(), B.contiguous(),
                                           C.contiguous(), D.contiguous(), delta_bias, True, True)

            outs.append(rearrange(out, "b d l -> b l d"))

        outs = self.multi_scan.multi_reverse(outs)
        outs = [self.attn(out) for out in outs]
        out = self.multi_scan(outs)
        out = out * z
        out = self.out_proj(out)

        return out


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


from torch_scatter import scatter_mean

# class GNNLocalCluster(nn.Module):
#     def __init__(self, dim, w_size=7, k_neighbors=9):
#         super().__init__()
#         self.dim = dim
#         self.w_size = w_size
#         self.k = k_neighbors  # 每个节点的邻居数
#
#         # 单一路径特征变换层
#         self.f = nn.Conv2d(dim, dim // 16, kernel_size=1)
#         self.p = nn.Conv2d(dim // 16, dim, kernel_size=1)
#
#         # 动态边权重参数
#         self.edge_alpha = nn.Parameter(torch.ones(1))
#         self.edge_beta = nn.Parameter(torch.zeros(1))
#
#     def build_graph(self, x):
#         """将单个样本特征图转换为图结构"""
#         # 输入 x: [1, C, H, W]
#         B, C, H, W = x.shape
#         assert B == 1, "build_graph expects batch size 1 per call"
#
#         x_flat = x.view(C, -1).permute(1, 0)  # [N, C] N=H*W
#
#         # 计算余弦相似度矩阵 [N, N]
#         sim = torch.cosine_similarity(
#             x_flat.unsqueeze(1),  # [N, 1, C]
#             x_flat.unsqueeze(0),  # [1, N, C]
#             dim=-1
#         )
#
#         # 找每个节点的 k 个邻居
#         _, topk_idx = sim.topk(self.k + 1, dim=1)  # k+1 包含自己
#         topk_idx = topk_idx[:, 1:]  # 去掉自己，变成 [N, k]
#
#         # 生成边索引
#         row = torch.arange(H * W, device=x.device).repeat_interleave(self.k)  # [N*k]
#         col = topk_idx.reshape(-1)  # [N*k]
#         edge_index = torch.stack([row, col], dim=0)  # [2, N*k]
#
#         return x_flat, edge_index
#
#     def forward(self, x_in):
#         # x_in: [B, C, H, W]
#         B, C, H, W = x_in.shape
#         # 1. 分块处理
#         x = rearrange(x_in, "b e (Wg w) (Hg h) -> (b Wg Hg) e w h", Wg=self.w_size, Hg=self.w_size)
#         out_list = []
#
#         # 逐个图处理
#         for i in range(x.shape[0]):
#             x_i = x[i:i + 1]  # [1, C, w, h]
#
#             # 2. 单路径特征变换
#             f = self.f(x_i)  # [1, C', w, h]
#
#             # 3. 构建图结构
#             nodes, edge_index = self.build_graph(f)  # nodes: [N, C'], edge_index: [2, E]
#
#             # 4. 图消息传递
#             msg = nodes[edge_index[1]]  # [E, C']
#             weights = torch.sigmoid(
#                 self.edge_beta + self.edge_alpha *
#                 torch.cosine_similarity(nodes[edge_index[0]], msg, dim=-1)
#             )
#             out = scatter_mean(msg * weights.unsqueeze(-1), edge_index[0], dim=0)  # [N, C']
#
#             # 5. 恢复空间结构
#             out = out.view(f.shape[-2], f.shape[-1], -1).permute(2, 0, 1).unsqueeze(0)  # [1, C', w, h]
#             out_list.append(out)
#
#         # 拼回 batch
#         out = torch.cat(out_list, dim=0)  # [(b*Wg*Hg), C', w, h]
#         # 逆变换恢复大图空间维度
#         out = rearrange(out, "(b Wg Hg) c w h -> b c (Wg w) (Hg h)", Wg=self.w_size, Hg=self.w_size, b=B)
#
#         # 6. 最终投影
#         out = self.p(out)  # [B, C, H, W]
#
#         # 返回展平后的空间维度
#         return out.flatten(2)  # [B, C, H*W]


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add, scatter_softmax
from einops import rearrange


class EnhancedGNNLocalCluster(nn.Module):
    def __init__(self, dim, w_size=7, k_neighbors=9):
        super().__init__()
        self.dim = dim
        self.w_size = w_size
        self.k = k_neighbors

        # 特征变换模块 (降维后维度为dim//4以获得更多中间特征)
        self.f = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GroupNorm(1, dim // 4)
        )
        self.p = nn.Sequential(
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.GroupNorm(1, dim)
        )

        # 残差路径
        self.residual = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GroupNorm(1, dim)
        )

        # 边权重计算MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * (dim // 4), dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()  # 输出0-1之间的权重
        )

        # 可学习的缩放参数
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def build_graph_batch(self, x):
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.view(B, C, N).permute(0, 2, 1)  # [B, N, C]

        # 空间坐标特征
        grid = torch.stack(torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device)
        ), dim=-1).float().view(N, 2)
        grid = (grid - grid.mean(0)) / (grid.std(0) + 1e-5)

        # 结合特征和空间信息
        x_aug = torch.cat([
            x_flat,
            grid.unsqueeze(0).expand(B, -1, -1)
        ], dim=-1)  # [B, N, C+2]

        # 计算增强后的相似度
        sim = F.cosine_similarity(
            x_aug.unsqueeze(2),
            x_aug.unsqueeze(1),
            dim=-1
        )  # [B, N, N]

        # 计算相似度后，将对角线置为负无穷
        mask = torch.eye(N, device=x.device, dtype=torch.bool)  # [N,N]单位矩阵
        sim.masked_fill_(mask, float('-inf'))  # 自身相似度设为-∞
        _, topk_idx = sim.topk(self.k, dim=-1)

        # 构建edge_index
        row = torch.arange(N, device=x.device).repeat(B, self.k, 1).transpose(1, 2).reshape(-1)
        col = topk_idx.reshape(-1)
        batch_offset = (torch.arange(B, device=x.device) * N).view(-1, 1, 1).expand(-1, N, self.k).reshape(-1)
        edge_index = torch.stack([row + batch_offset, col + batch_offset], dim=0)

        x_flat_all = x_flat.reshape(B * N, C)
        return x_flat_all, edge_index, B, H, W

    def forward(self, x_in):
        B, C, H, W = x_in.shape
        assert H % self.w_size == 0 and W % self.w_size == 0

        # 残差路径
        residual = self.residual(x_in)

        # 分块处理
        x = rearrange(x_in, "b c (Wg w) (Hg h) -> (b Wg Hg) c w h",
                      Wg=self.w_size, Hg=self.w_size)

        # 特征变换 + 归一化
        f = self.f(x)

        # 图构建
        nodes, edge_index, B_patch, H_patch, W_patch = self.build_graph_batch(f)
        msg = nodes[edge_index[1]]

        # 改进的边权重计算
        edge_feat = torch.cat([
            nodes[edge_index[0]],
            nodes[edge_index[1]]
        ], dim=-1)
        weights = self.edge_mlp(edge_feat).squeeze(-1)

        # 加权聚合 + 门控机制
        out = scatter_add(
            msg * weights.unsqueeze(-1),
            edge_index[0],
            dim=0
        )
        out = out * self.gamma + nodes * self.beta  # 可学习的混合比例

        # 空间还原
        out = out.view(B_patch, H_patch, W_patch, -1).permute(0, 3, 1, 2)
        out = rearrange(out, "(b Wg Hg) c w h -> b c (Wg w) (Hg h)",
                        Wg=self.w_size, Hg=self.w_size, b=B)

        # 最终投影 + 残差连接
        out = self.p(out) + residual
        return out


class GNNLocalCluster0(nn.Module):
    def __init__(self, dim, w_size=7, k_neighbors=9):
        super().__init__()
        self.dim = dim
        self.w_size = w_size
        self.k = k_neighbors

        # 特征变换模块
        self.f = nn.Conv2d(dim, dim // 16, kernel_size=1)
        self.p = nn.Conv2d(dim // 16, dim, kernel_size=1)

        # 边权重参数
        self.edge_alpha = nn.Parameter(torch.ones(1))
        self.edge_beta = nn.Parameter(torch.zeros(1))

    def build_graph_batch(self, x):
        # x: [B_patch, C, H_patch, W_patch]
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.view(B, C, N).permute(0, 2, 1)  # [B, N, C]

        # 计算余弦相似度矩阵 [B, N, N]
        sim = F.cosine_similarity(x_flat.unsqueeze(2), x_flat.unsqueeze(1), dim=-1)  # [B, N, N]
        _, topk_idx = sim.topk(self.k, dim=-1)

        # 构建 edge_index
        row = torch.arange(N, device=x.device).repeat(B, self.k, 1).transpose(1, 2).reshape(-1)  # [B*N*k]
        col = topk_idx.reshape(-1)  # [B*N*k]

        batch_offset = (torch.arange(B, device=x.device) * N).view(-1, 1, 1).expand(-1, N, self.k).reshape(-1)
        edge_index = torch.stack([row + batch_offset, col + batch_offset], dim=0)  # [2, B*N*k]

        x_flat_all = x_flat.reshape(B * N, C)  # [B*N, C]
        return x_flat_all, edge_index, B, H, W

    def forward(self, x_in):
        # x_in: [B, C, H, W]
        B, C, H, W = x_in.shape
        assert H % self.w_size == 0 and W % self.w_size == 0, "输入尺寸必须能被 w_size 整除"

        # 分块处理 -> [B*Wg*Hg, C, w, h]
        x = rearrange(x_in, "b c (Wg w) (Hg h) -> (b Wg Hg) c w h", Wg=self.w_size, Hg=self.w_size)

        # 特征变换 -> [B_patch, C']
        f = self.f(x)  # [B*, C', w, h]

        # 图构建 + 聚合操作
        nodes, edge_index, B_patch, H_patch, W_patch = self.build_graph_batch(f)  # [B*N, C']
        msg = nodes[edge_index[1]]  # [E, C']
        weights = torch.sigmoid(
            self.edge_beta + self.edge_alpha * F.cosine_similarity(nodes[edge_index[0]], msg, dim=-1)
        )
        # out = scatter_mean(msg * weights.unsqueeze(-1), edge_index[0], dim=0)  # [B*N, C']
        # out = out + nodes
        weight_sum = scatter_add(weights, edge_index[0], dim=0)  # [B*N]
        eps = 1e-12
        weights_norm = weights / (weight_sum[edge_index[0]] + eps)

        # 加权聚合
        out = scatter_add(msg * weights_norm.unsqueeze(-1), edge_index[0], dim=0)  # [B*N, C']

        # 空间还原 -> [B_patch, C', H_patch, W_patch]
        C_ = out.shape[-1]
        out = out.view(B_patch, H_patch, W_patch, C_).permute(0, 3, 1, 2)  # [B, C', H, W]

        # 合并小块图 -> [B, C', H, W]
        out = rearrange(out, "(b Wg Hg) c w h -> b c (Wg w) (Hg h)", Wg=self.w_size, Hg=self.w_size, b=B)

        # 通道投影恢复维度
        out = self.p(out)  # [B, C, H, W]
        return out.flatten(2)  # [B, C, H*W]


# class GNNLocalCluster0_F(nn.Module):
#     def __init__(self, dim, w_size=7, k_neighbors=9):
#         super().__init__()
#         self.dim = dim
#         self.w_size = w_size
#         self.k = k_neighbors
#
#         # 特征变换模块
#         self.f = nn.Conv2d(dim, dim // 8, kernel_size=1)
#         self.p = nn.Conv2d(dim // 8, dim, kernel_size=1)
#
#         # 边权重参数
#         self.edge_alpha = nn.Parameter(torch.ones(1))
#         self.edge_beta = nn.Parameter(torch.zeros(1))
#
#     def build_graph_batch(self, x):
#         # x: [B_patch, C, H_patch, W_patch]
#         B, C, H, W = x.shape
#         N = H * W
#         x_flat = x.view(B, C, N).permute(0, 2, 1)  # [B, N, C]
#
#         # 空间坐标特征
#         grid = torch.stack(torch.meshgrid(
#             torch.arange(H, device=x.device),
#             torch.arange(W, device=x.device)
#         ), dim=-1).float().view(N, 2)
#         grid = (grid - grid.mean(0)) / (grid.std(0) + 1e-5)
#
#         # 结合特征和空间信息
#         x_aug = torch.cat([
#             x_flat,
#             grid.unsqueeze(0).expand(B, -1, -1)
#         ], dim=-1)  # [B, N, C+2]
#
#         # 计算增强后的相似度
#         sim = F.cosine_similarity(
#             x_aug.unsqueeze(2),
#             x_aug.unsqueeze(1),
#             dim=-1
#         )  # [B, N, N]
#
#         #
#         # # 计算余弦相似度矩阵 [B, N, N]
#         # sim = F.cosine_similarity(x_flat.unsqueeze(2), x_flat.unsqueeze(1), dim=-1)  # [B, N, N]
#         _, topk_idx = sim.topk(self.k, dim=-1)
#
#         # 构建 edge_index
#         row = torch.arange(N, device=x.device).repeat(B, self.k, 1).transpose(1, 2).reshape(-1)  # [B*N*k]
#         col = topk_idx.reshape(-1)  # [B*N*k]
#
#         batch_offset = (torch.arange(B, device=x.device) * N).view(-1, 1, 1).expand(-1, N, self.k).reshape(-1)
#         edge_index = torch.stack([row + batch_offset, col + batch_offset], dim=0)  # [2, B*N*k]
#
#         x_flat_all = x_flat.reshape(B * N, C)  # [B*N, C]
#         return x_flat_all, edge_index, B, H, W
#
#     def forward(self, x_in):
#         # x_in: [B, C, H, W]
#         B, C, H, W = x_in.shape
#         assert H % self.w_size == 0 and W % self.w_size == 0, "输入尺寸必须能被 w_size 整除"
#
#         # 分块处理 -> [B*Wg*Hg, C, w, h]
#         x = rearrange(x_in, "b c (Wg w) (Hg h) -> (b Wg Hg) c w h", Wg=self.w_size, Hg=self.w_size)
#
#         # 特征变换 -> [B_patch, C']
#         f = self.f(x)  # [B*, C', w, h]
#
#         # 图构建 + 聚合操作
#         nodes, edge_index, B_patch, H_patch, W_patch = self.build_graph_batch(f)  # [B*N, C']
#         msg = nodes[edge_index[1]]  # [E, C']
#         weights = torch.sigmoid(
#             self.edge_beta + self.edge_alpha * F.cosine_similarity(nodes[edge_index[0]], msg, dim=-1)
#         )
#         # out = scatter_mean(msg * weights.unsqueeze(-1), edge_index[0], dim=0)  # [B*N, C']
#         # out = out + nodes
#         weight_sum = scatter_add(weights, edge_index[0], dim=0)  # [B*N]
#         eps = 1e-12
#         weights_norm = weights / (weight_sum[edge_index[0]] + eps)
#
#         # 加权聚合
#         out = scatter_add(msg * weights_norm.unsqueeze(-1), edge_index[0], dim=0)  # [B*N, C']
#
#         # 空间还原 -> [B_patch, C', H_patch, W_patch]
#         C_ = out.shape[-1]
#         out = out.view(B_patch, H_patch, W_patch, C_).permute(0, 3, 1, 2)  # [B, C', H, W]
#
#         # 合并小块图 -> [B, C', H, W]
#         out = rearrange(out, "(b Wg Hg) c w h -> b c (Wg w) (Hg h)", Wg=self.w_size, Hg=self.w_size, b=B)
#
#         # 通道投影恢复维度
#         out = self.p(out)
#
#         return out

import matplotlib.pyplot as plt


def visualize_connections(patch, edge_index, H, W):
    """可视化一个patch的连接关系"""
    plt.figure(figsize=(10, 10))

    # 绘制节点位置
    grid = torch.stack(torch.meshgrid(
        torch.arange(H), torch.arange(W)
    )).permute(1, 2, 0)

    plt.scatter(grid[:, :, 1], grid[:, :, 0], s=100, c='blue')

    # 绘制边
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
    tgt = edge_index[1, i].item()

    src_y, src_x = divmod(src, W)
    tgt_y, tgt_x = divmod(tgt, W)

    plt.plot([src_x, tgt_x], [src_y, tgt_y], 'r-', alpha=0.3)

    plt.title("Patch内节点连接关系")
    plt.gca().invert_yaxis()  # 保持图像坐标系
    plt.show()


class GNNLocalCluster0_F(nn.Module):
    def __init__(self, dim, w_size=7, k_neighbors=15):
        super().__init__()
        self.dim = dim
        self.w_size = w_size
        self.k = k_neighbors
        self.discuss = []
        self.discuss_weights = []
        # 特征变换模块
        # self.f = nn.Conv2d(dim, dim // 16, kernel_size=1)
        # self.p = nn.Conv2d(dim // 16, dim, kernel_size=1)
        self.f = nn.Sequential(
            nn.Conv2d(dim, dim // 8, kernel_size=1),
        )
        self.p = nn.Sequential(
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )

        # 边权重参数
        self.edge_alpha = nn.Parameter(torch.ones(1))
        self.edge_beta = nn.Parameter(torch.zeros(1))

    def build_graph_batch(self, x):
        # x: [B_patch, C, H_patch, W_patch]
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.view(B, C, N).permute(0, 2, 1)  # [B, N, C]
        # print(x.shape, x_flat.shape) torch.Size([64, 8, 8, 8]) torch.Size([64, 64, 8])

        # 空间坐标特征
        grid = torch.stack(torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device)
        ), dim=-1).float().view(N, 2)
        grid = (grid - grid.mean(0)) / (grid.std(0) + 1e-5)
        # 结合特征和空间信息
        x_aug = torch.cat([
            x_flat,
            grid.unsqueeze(0).expand(B, -1, -1)
        ], dim=-1)  # [B, N, C+2]

        # 计算增强后的相似度
        sim = F.cosine_similarity(
            x_aug.unsqueeze(2),
            x_aug.unsqueeze(1),
            dim=-1
        )
        _, topk_idx = sim.topk(self.k, dim=-1)

        # 构建 edge_index
        row = torch.arange(N, device=x.device).repeat(B, self.k, 1).transpose(1, 2).reshape(-1)  # [B*N*k]
        col = topk_idx.reshape(-1)  # [B*N*k]

        batch_offset = (torch.arange(B, device=x.device) * N).view(-1, 1, 1).expand(-1, N, self.k).reshape(-1)
        edge_index = torch.stack([row + batch_offset, col + batch_offset], dim=0)  # [2, B*N*k]
        # print(x.shape, edge_index.shape, edge_index[0, :20], edge_index[1, :20])
        # np.save('edge_index.npy', edge_index.cpu().numpy())
        # self.discuss.append(edge_index.cpu().numpy())
        # if B > 0:
        #     first_batch_edges = edge_index[:, edge_index[0] < H * W]
        #     # print(edge_index.shape)
        #     visualize_connections(x[0], first_batch_edges, H, W)
        #     input("可视化完成，按Enter键继续...")  # 暂停直到用户按键

        x_flat_all = x_aug.reshape(B * N, C + 2)
        # print(x_aug.shape, x_flat_all.shape) torch.Size([64, 64, 10]) torch.Size([4096, 10])
        return x_flat_all, edge_index, B, H, W

    # def forward(self, x_in):
    #     # x_in: [B, C, H, W]
    #     B, C, H, W = x_in.shape
    #     assert H % self.w_size == 0 and W % self.w_size == 0, "输入尺寸必须能被 w_size 整除"
    #
    #     # 分块处理 -> [B*Wg*Hg, C, w, h]
    #     x = rearrange(x_in, "b c (Wg w) (Hg h) -> (b Wg Hg) c w h", Wg=self.w_size, Hg=self.w_size)
    #     # print(x.shape) torch.Size([1024, 64, 8, 8])
    #     # 特征变换 -> [B_patch, C']
    #     f = self.f(x)  # [B*, C', w, h]
    #
    #     # 图构建 + 聚合操作
    #     nodes, edge_index, B_patch, H_patch, W_patch = self.build_graph_batch(f)  # [B*N, C']
    #     msg = nodes[edge_index[1]]  # [E, C']
    #     weights = torch.sigmoid(
    #         self.edge_beta + self.edge_alpha * F.cosine_similarity(nodes[edge_index[0]], msg, dim=-1)
    #     )
    #     # weights = self.edge_beta + self.edge_alpha * F.cosine_similarity(nodes[edge_index[0]], msg, dim=-1)
    #     # out = scatter_mean(msg * weights.unsqueeze(-1), edge_index[0], dim=0)  # [B*N, C']
    #     # out = out + nodes
    #     # print(F.cosine_similarity(nodes[edge_index[0]], msg, dim=-1).shape, F.cosine_similarity(nodes[edge_index[0]], msg, dim=-1)[:30])
    #     # weight_sum = scatter_add(weights, edge_index[0], dim=0)
    #     # eps = 1e-12
    #     # weights_norm = weights / (weight_sum[edge_index[0]] + eps)
    #     weights_norm = scatter_softmax(weights, edge_index[0], dim=0)
    #     # print(F.cosine_similarity(nodes[edge_index[0]], msg, dim=-1)[:10], weights_norm[:10])
    #     # print(msg.shape, weights_norm.shape, nodes.shape) torch.Size([40960, 10]) torch.Size([40960]) torch.Size([4096, 10])
    #     # 加权聚合
    #     out = scatter_add(msg[:, :-2] * weights_norm.unsqueeze(-1), edge_index[0], dim=0)  # [B*N, C']
    #
    #     # 空间还原 -> [B_patch, C', H_patch, W_patch]
    #     C_ = out.shape[-1]
    #     out = out.view(B_patch, H_patch, W_patch, C_).permute(0, 3, 1, 2)  # [B, C', H, W]
    #
    #     # 合并小块图 -> [B, C', H, W]
    #     out = rearrange(out, "(b Wg Hg) c w h -> b c (Wg w) (Hg h)", Wg=self.w_size, Hg=self.w_size, b=B)
    #
    #     # 通道投影恢复维度
    #     out = self.p(out)
    #
    #     return out
    def forward(self, x_in):
        B, C, H, W = x_in.shape
        assert H % self.w_size == 0 and W % self.w_size == 0, "输入尺寸必须能被 w_size 整除"
        # 分块处理 -> [B*Wg*Hg, C, w, h]
        x_patches = rearrange(x_in, "b c (Wg w) (Hg h) -> (b Wg Hg) c w h", Wg=self.w_size, Hg=self.w_size)

        # 特征变换 -> [B*, C//8, w, h]
        f = self.f(x_patches)  # 降维到低维空间计算相似度
        # print(x_in.shape, x_patches.shape, f.shape)
        # 图构建 + 权重计算
        nodes, edge_index, B_patch, H_patch, W_patch = self.build_graph_batch(f)
        weights = torch.sigmoid(
            self.edge_beta + self.edge_alpha * F.cosine_similarity(nodes[edge_index[0]], nodes[edge_index[1]], dim=-1)
        )
        weights_norm = scatter_softmax(weights, edge_index[0], dim=0)
        # self.discuss_weights.append(weights.cpu().numpy())
        # print(weights.shape, weights_norm.shape, weights[:20], weights_norm[:20])

        # 对原始特征加权聚合（跳过 self.p）
        # 展平方式与 build_graph_batch 一致：先 view(B, C, N) 再 permute(0, 2, 1)
        x_patches_flat = x_patches.view(B_patch, C, H_patch * W_patch).permute(0, 2, 1)  # [B, N, C]
        x_patches_flat = x_patches_flat.reshape(-1, C)  # [B*N, C]（全局索引）
        msg_original = x_patches_flat[edge_index[1]]  # [E, C]
        weighted_msg = msg_original * weights_norm.unsqueeze(-1)  # [E, C]
        out_flat = scatter_add(weighted_msg, edge_index[0], dim=0)  # [B*N, C]

        # 还原空间维度
        out = out_flat.view(B_patch, H_patch, W_patch, C).permute(0, 3, 1, 2)  # [B*, C, H, W]
        out = rearrange(out, "(b Wg Hg) c w h -> b c (Wg w) (Hg h)", Wg=self.w_size, Hg=self.w_size, b=B)

        return out  # [B, C, H, W]


class GNNLocalCluster(nn.Module):
    def __init__(self, dim, w_size=7, k_neighbors=9, sigma=4.0):
        super().__init__()
        self.dim = dim
        self.w_size = w_size
        self.k = k_neighbors
        # self.sigma = sigma
        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # 特征变换模块
        self.f = nn.Conv2d(dim, dim // 4, kernel_size=1)
        self.p = nn.Conv2d(dim // 4, dim, kernel_size=1)

        # 边权重参数（可训练）
        self.edge_alpha_feat = nn.Parameter(torch.ones(1))
        self.edge_alpha_dist = nn.Parameter(torch.ones(1))
        self.edge_beta = nn.Parameter(torch.zeros(1))

        self.edge_mlp = nn.Sequential(
            nn.Linear(2, 4),
            nn.SiLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def build_graph_batch(self, x):
        """
        构建图，结合特征相似度和空间距离选邻居
        x: [B_patch, C, H_patch, W_patch]
        """
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.view(B, C, N).permute(0, 2, 1)  # [B, N, C]

        # 计算特征余弦相似度矩阵 [B, N, N]
        sim_feat = F.cosine_similarity(x_flat.unsqueeze(2), x_flat.unsqueeze(1), dim=-1)

        # 计算空间距离矩阵
        device = x.device
        coords = torch.stack(torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device)),
                             dim=-1)  # [H, W, 2]
        coords = coords.view(-1, 2).float()  # [N, 2]

        dist_mat = torch.cdist(coords, coords, p=2)  # [N, N]
        dist_mat = dist_mat.unsqueeze(0).repeat(B, 1, 1)  # [B, N, N]

        # 距离相似度，高斯核
        sim_dist = torch.exp(- dist_mat ** 2 / (2 * self.sigma ** 2))  # [B, N, N]

        # 结合相似度，注意这里简单相乘，也可以用加权和
        # combined_sim = sim_feat * sim_dist  # [B, N, N]
        combined_sim = self.alpha * sim_feat + (1 - self.alpha) * sim_dist

        # 选top k邻居，包含自身节点（保证自身在内）
        _, topk_idx = combined_sim.topk(self.k, dim=-1)  # [B, N, k]

        # 构建 edge_index
        row = torch.arange(N, device=device).repeat(B, self.k, 1).transpose(1, 2).reshape(-1)  # [B*N*k]
        col = topk_idx.reshape(-1)  # [B*N*k]

        batch_offset = (torch.arange(B, device=device) * N).view(-1, 1, 1).expand(-1, N, self.k).reshape(-1)
        edge_index = torch.stack([row + batch_offset, col + batch_offset], dim=0)  # [2, B*N*k]

        x_flat_all = x_flat.reshape(B * N, C)  # [B*N, C]
        return x_flat_all, edge_index, B, H, W, coords

    def forward(self, x_in):
        """
        x_in: [B, C, H, W]
        """
        B, C, H, W = x_in.shape
        assert H % self.w_size == 0 and W % self.w_size == 0, "输入尺寸必须能被 w_size 整除"

        # 分块处理 -> [B*Wg*Hg, C, w, h]
        x = rearrange(x_in, "b c (Wg w) (Hg h) -> (b Wg Hg) c w h", Wg=self.w_size, Hg=self.w_size)

        # 特征变换 -> [B_patch, C', w, h]
        f = self.f(x)
        # f = x

        # 构建图，获得邻居索引和节点特征
        nodes, edge_index, B_patch, H_patch, W_patch, coords = self.build_graph_batch(f)  # nodes: [B*N, C']

        # 取邻居消息
        msg = nodes[edge_index[1]]  # [E, C']

        # 计算特征余弦相似度权重
        sim_feat_edge = F.cosine_similarity(nodes[edge_index[0]], msg, dim=-1)

        # 计算空间距离权重
        # 计算对应节点的coords索引（注意batch offset）
        N = H_patch * W_patch
        device = nodes.device
        coords = coords.to(device)  # [N, 2]

        src_idx = (edge_index[0] % N).long()
        tgt_idx = (edge_index[1] % N).long()

        src_coords = coords[src_idx]  # [E, 2]
        tgt_coords = coords[tgt_idx]  # [E, 2]
        dist = torch.norm(src_coords - tgt_coords, dim=-1)

        sim_dist_edge = torch.exp(- dist ** 2 / (2 * self.sigma ** 2))

        # 综合计算权重
        # weights = torch.sigmoid(
        #     self.edge_beta
        #     + self.edge_alpha_feat * sim_feat_edge
        #     + self.edge_alpha_dist * sim_dist_edge
        # )  # [E]
        sim_pair = torch.stack([sim_feat_edge, sim_dist_edge], dim=-1)  # [E, 2]
        weights = self.edge_mlp(sim_pair).squeeze(-1)

        # 权重归一化（确保每个节点的邻居权重和为1）
        weight_sum = scatter_add(weights, edge_index[0], dim=0)  # [B*N]
        eps = 1e-12
        weights_norm = weights / (weight_sum[edge_index[0]] + eps)

        # 加权聚合
        out = scatter_add(msg * weights_norm.unsqueeze(-1), edge_index[0], dim=0)  # [B*N, C']

        # 空间还原 -> [B_patch, C', H_patch, W_patch]
        C_ = out.shape[-1]
        out = out.view(B_patch, H_patch, W_patch, C_).permute(0, 3, 1, 2)

        # 合并小块图 -> [B, C', H, W]
        out = rearrange(out, "(b Wg Hg) c w h -> b c (Wg w) (Hg h)", Wg=self.w_size, Hg=self.w_size,
                        b=B_patch // (self.w_size ** 2))

        # 通道投影恢复维度
        out = self.p(out)  # [B, C, H, W]

        # 返回 [B, C, H*W]
        return out.flatten(2)


class LocalCluster(nn.Module):
    def __init__(self, dim, w_size=7, clusters=7):
        super().__init__()
        self.dim = dim
        self.w_size = w_size
        self.clusters = clusters
        self.centers_proposal = nn.AdaptiveAvgPool2d((self.clusters, self.clusters))

        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))

        self.f = nn.Conv2d(self.dim // 2, self.dim // 16, kernel_size=1)
        self.v = nn.Conv2d(self.dim // 2, self.dim // 16, kernel_size=1)
        self.p = nn.Conv2d(self.dim // 16, self.dim, kernel_size=1)

    def cluster(self, f, v, Wg, Hg):
        bb, cc, ww, hh = f.shape
        # print(f.shape, "x") torch.Size([64, 12, 8, 8]) x
        centers = self.centers_proposal(f)
        # print(centers.shape, "q") torch.Size([64, 12, 5, 5]) q
        value_centers = rearrange(self.centers_proposal(v), 'b c w h -> b (w h) c')
        # print(value_centers.shape, "y") torch.Size([64, 25, 12]) y
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(bb, cc, -1).permute(0, 2, 1),
                f.reshape(bb, cc, -1).permute(0, 2, 1)
            )
        )
        # print(sim.shape, centers.reshape(bb, cc, -1).permute(0, 2, 1).shape, f.reshape(bb, cc, -1).permute(0, 2, 1).shape)
        # torch.Size([64, 25, 64]) torch.Size([64, 25, 12])  torch.Size([64, 64, 12])

        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        v2 = rearrange(v, 'b c w h -> b (w h) c')
        # print(sim.shape, v2.shape) torch.Size([64, 25, 64]) torch.Size([64, 64, 12])

        out = ((v2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                sim.sum(dim=-1, keepdim=True) + 1.0)
        # print(out.shape, 'x') torch.Size([64, 25, 12]) x
        # dispatch step, return to each point in a cluster
        out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)
        # print(out.shape, 'x1') torch.Size([64, 64, 12]) x1
        out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        # print(out.shape, 'x2') torch.Size([64, 12, 8, 8]) x2
        out = rearrange(out, "(b Wg Hg) e w h-> b e (Wg w) (Hg h)", Wg=Wg, Hg=Hg)
        # print(out.shape, 'x3') torch.Size([1, 12, 64, 64]) x3
        return out

    def forward(self, x_in):
        x = rearrange(x_in, "b e (Wg w) (Hg h)-> (b Wg Hg) e w h", Wg=self.w_size, Hg=self.w_size)
        x1, x2 = x.chunk(2, dim=1)
        f = self.f(x1)
        v = self.v(x2)
        # print(f.shape, v.shape) torch.Size([64, 12, 8, 8]) torch.Size([64, 12, 8, 8])
        out = self.cluster(f, v, self.w_size, self.w_size)
        # print(out.shape, "1") torch.Size([1, 12, 64, 64]) 1
        out = self.p(out)
        # print(out.shape, "2") torch.Size([1, 192, 64, 64]) 2
        out = out.flatten(2)
        # print(out.shape, "3") torch.Size([1, 192, 4096]) 3
        return out


# class LocalCluster(nn.Module):
#     def __init__(self, dim, w_size=7, clusters=5):
#         super().__init__()
#         self.dim = dim
#         self.w_size = w_size
#         self.clusters = clusters
#         self.centers_proposal = nn.AdaptiveAvgPool2d((self.clusters, self.clusters))
#
#         self.sim_alpha = nn.Parameter(torch.ones(1))
#         self.sim_beta = nn.Parameter(torch.zeros(1))
#
#         self.f = nn.Conv2d(self.dim // 2, self.dim // 8, kernel_size=1)
#         self.v = nn.Conv2d(self.dim // 2, self.dim // 8, kernel_size=1)
#         self.p = nn.Conv2d(self.dim // 8, self.dim, kernel_size=1)
#
#     def cluster(self, f, v, Wg, Hg):
#         bb, cc, ww, hh = f.shape
#         centers = self.centers_proposal(f)
#         value_centers = rearrange(self.centers_proposal(v), 'b c w h -> b (w h) c')  # [b,C_W,C_H,c] [32768,4,24]
#
#         sim = torch.sigmoid(
#             self.sim_beta +
#             self.sim_alpha * pairwise_cos_sim(
#                 centers.reshape(bb, cc, -1).permute(0, 2, 1),  # [32768,24,2,2]---[32768,4,24] 4是中心点，24是特征
#                 f.reshape(bb, cc, -1).permute(0, 2, 1)  # [32768,24,7,7]---[32768,49,24]  49是块中的所有点，24是特征
#             )
#         )
#         sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
#         mask = torch.zeros_like(sim)
#         mask.scatter_(1, sim_max_idx, 1.)
#         sim = sim * mask
#         v2 = rearrange(v, 'b c w h -> b (w h) c')
#         out = ((v2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
#                 sim.sum(dim=-1, keepdim=True) + 1.0)
#
#         # dispatch step, return to each point in a cluster
#         out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D] [32768,49,24]
#         out = rearrange(out, "b (w h) c -> b c w h", w=ww)  # [32768,24,7,7]
#         out = rearrange(out, "(b Wg Hg) e w h-> b e (Wg w) (Hg h)", Wg=Wg, Hg=Hg)
#
#         return out
#
#     def forward(self, x_in):
#         x = rearrange(x_in, "b e (Wg w) (Hg h)-> (b Wg Hg) e w h", Wg=self.w_size, Hg=self.w_size)
#
#         x1, x2 = x.chunk(2, dim=1)
#         f = self.f(x1)
#         v = self.v(x2)
#         out = self.cluster(f, v, self.w_size, self.w_size)
#         out = self.p(out)
#
#         out = out.flatten(2)
#
#         return out


def fea_num_reverse(x, H, W):
    K = 7
    if len(x.shape) == 4:
        B, C, H, W = x.shape
    elif len(x.shape) == 3:
        B, C, _ = x.shape
        assert H is not None and W is not None, "x must be BCHW format to infer the H W"
    else:
        raise RuntimeError(f"Unsupported shape of x: {x.shape}")
    B, C, H, W = int(B), int(C), int(H), int(W)

    Hg, Wg = math.ceil(H / K), math.ceil(W / K)
    Hb, Wb = Hg * K, Wg * K

    x = rearrange(x, "b c (w h ) -> b c w h", w=Wb, h=Hb)
    x = x[:, :, :W, :H]
    x = x.flatten(2)

    return x
