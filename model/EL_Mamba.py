import math
from functools import partial
from typing import Callable, Any
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
# from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from model.multi_mamba import MultiScan

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

try:
    "sscore acts the same as mamba_ssm"
    SSMODE = "sscore"
    if torch.__version__ > '2.0.0':
        from selective_scan_vmamba_pt202 import selective_scan_cuda_core
    else:
        from selective_scan_vmamba import selective_scan_cuda_core
except Exception as e:
    print(e, "you should install mamba_ssm to use this", flush=True)
    SSMODE = "mamba_ssm"
    import selective_scan_cuda
    # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


# SSMODE = "mamba_ssm"
# import selective_scan_cuda

# fvcore flops =======================================

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def selective_scan_flop_jit(inputs, outputs):
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False, with_Group=True)
    return flops


class SelectiveScan(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}"  # 8+ is too slow to compile
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True

        if SSMODE == "mamba_ssm":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        else:
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        if SSMODE == "mamba_ssm":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
                False  # option to recompute out_z, not used here
            )
        else:
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )

        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


"""
Local Mamba
"""


class MultiScanVSSM(MultiScan):
    ALL_CHOICES = MultiScan.ALL_CHOICES

    def __init__(self, dim, choices=None, win_size=8):
        super().__init__(dim, choices=choices, token_size=None, win_size=win_size)
        self.attn = BiAttn(dim)

    def merge(self, xs):
        # xs: [B, K, D, L]
        # return: [B, D, L]

        # remove the padded tokens
        xs = [xs[:, i, :, :l] for i, l in enumerate(self.scan_lengths)]
        xs = super().multi_reverse(xs)
        # xs = [self.attn(x.transpose(-2, -1)) for x in xs]
        xs = [x.transpose(-2, -1) for x in xs]
        x = super().forward(xs)

        return x

    def multi_scan(self, x):
        # x: [B, C, H, W]
        # return: [B, K, C, H * W]
        B, C, H, W = x.shape
        self.token_size = (H, W)

        xs = super().multi_scan(x)  # [[B, C, L], ...]
        self.scan_lengths = [x.shape[2] for x in xs]
        max_length = max(self.scan_lengths)

        # pad the tokens into the same length as VMamba compute all directions together
        new_xs = []
        for x in xs:
            if x.shape[2] < max_length:
                x = F.pad(x, (0, max_length - x.shape[2]))
            new_xs.append(x)

        return torch.stack(new_xs, 1)

    def __repr__(self):
        scans = ', '.join(self.choices)
        return super().__repr__().replace('MultiScanVSSM', f'MultiScanVSSM[{scans}]')


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
        # s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(-1, x.shape[1], -1)], dim=-1))
        # s_attn = self.gate_fn(s_attn)  # [B, N, 1]

        attn = c_attn  # * s_attn  # [B, N, C]
        out = ori_x * attn
        return out


def multi_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        nrows=-1,
        delta_softplus=True,
        to_dtype=True,
        multi_scan=None,
        win_size=8,
):
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1
    # print(x.shape) torch.Size([1, 192, 64, 64])
    xs = multi_scan.multi_scan(x)

    # print(xs.shape) torch.Size([1, 4, 192, 4096])
    L = xs.shape[-1]
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)  # l fixed
    # print(x_dbl.shape) torch.Size([1, 4, 38, 4096])
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    # print(dts.shape, R, N) torch.Size([1, 4, 6, 4096]) 6 16
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L).to(torch.float)
    dts = dts.contiguous().view(B, -1, L).to(torch.float)
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous().to(torch.float)
    Cs = Cs.contiguous().to(torch.float)
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    ).view(B, K, -1, L)

    y = multi_scan.merge(ys)

    y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            simple_init=False,
            directions=None,
            win_size=8,
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv

        self.out_norm = nn.LayerNorm(d_inner)
        self.K = len(MultiScanVSSM.ALL_CHOICES) if directions is None else len(directions)
        self.K2 = self.K

        # in proj =======================================
        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)  # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.win_size = win_size
        # Local Mamba
        self.multi_scan = MultiScanVSSM(d_expand, choices=directions, win_size=self.win_size)

        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((self.K2 * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, nrows=-1, channel_first=False):
        # print(self.ssm_low_rank   False)
        nrows = 1
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = multi_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, self.out_norm,
            nrows=nrows, delta_softplus=True, multi_scan=self.multi_scan, win_size=self.win_size
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1)
            z = self.act(z)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x))
        else:
            xz = self.act(xz)
            x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_simple_init=False,
            win_size=8,
            # =============================
            use_checkpoint: bool = False,
            directions=None,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            # bias=False,
            # ==========================
            # dt_min=0.001,
            # dt_max=0.1,
            # dt_init="random",
            # dt_scale="random",
            # dt_init_floor=1e-4,
            simple_init=ssm_simple_init,
            # ==========================
            directions=directions,
            win_size=win_size,
        )
        self.drop_path = DropPath(drop_path)

    def _forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.op(self.norm(input)))
        # print(x.shape, "YYY")
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def forward(self, x):  # ResidualBlock的前向传播函数 （29）
        out = self.left(x)
        if self.right is None:
            residual = x
        else:
            residual = self.right(x)
        out += residual
        return F.relu(out)


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


# class CrossAttention(nn.Module):
#     def __init__(self, dim_q, dim_kv, num_heads=8):
#         super().__init__()
#         self.num_heads = num_heads
#         self.dim_q = dim_q
#         self.dim_kv = dim_kv
#         self.head_dim = dim_q // num_heads
#
#         self.query_proj = nn.Linear(dim_q, dim_q)
#         self.key_proj = nn.Linear(dim_kv, dim_q)
#         self.value_proj = nn.Linear(dim_kv, dim_q)
#
#         self.out_proj = nn.Linear(dim_q, dim_q)
#
#     def forward(self, query, key_value):
#         # query: (B, HW, C), key_value: (B, N, D)
#         B, Q_len, _ = query.shape
#         K_len = key_value.shape[1]
#
#         # Linear projections
#         Q = self.query_proj(query).view(B, Q_len, self.num_heads, self.head_dim).transpose(1,
#                                                                                            2)  # (B, heads, Q_len, head_dim)
#         K = self.key_proj(key_value).view(B, K_len, self.num_heads, self.head_dim).transpose(1, 2)
#         V = self.value_proj(key_value).view(B, K_len, self.num_heads, self.head_dim).transpose(1, 2)
#         # print(Q.shape, K.shape, V.shape)
#         # Scaled Dot-Product Attention
#         scores = (Q @ K.transpose(-2, -1)) / self.head_dim ** 0.5  # (B, heads, Q_len, K_len)
#         attn = F.softmax(scores, dim=-1)
#         out = attn @ V  # (B, heads, Q_len, head_dim)
#
#         out = out.transpose(1, 2).contiguous().view(B, Q_len, self.dim_q)
#         return self.out_proj(out)


class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.head_dim = dim_q // num_heads

        assert dim_q % num_heads == 0, "dim_q must be divisible by num_heads"

        # 线性映射层
        self.query_proj = nn.Linear(dim_q, dim_q)
        self.key_proj = nn.Linear(dim_kv, dim_q)
        self.value_proj = nn.Linear(dim_kv, dim_q)
        self.out_proj = nn.Linear(dim_q, dim_q)

        # LayerNorm + Dropout
        self.norm = nn.LayerNorm(dim_q)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, query, key_value):
        # query: (B, Q_len, dim_q)
        # key_value: (B, K_len, dim_kv)
        B, Q_len, _ = query.shape
        K_len = key_value.shape[1]

        residual = query  # 残差连接
        # Linear projections
        Q = self.query_proj(query).view(B, Q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key_value).view(B, K_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(key_value).view(B, K_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)  # 加 Dropout
        out = attn @ V

        # 合并 heads
        out = out.transpose(1, 2).contiguous().view(B, Q_len, self.dim_q)
        out = self.out_proj(out)
        out = self.proj_drop(out)  # 输出也加 Dropout

        # 残差连接 + LayerNorm
        out = self.norm(out + residual)

        return out


class Conv_Extra(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   # build_norm_layer(norm_layer, 64)[1],
                                   nn.BatchNorm2d(64),
                                   act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   # build_norm_layer(norm_layer, 64)[1],
                                   nn.BatchNorm2d(64),
                                   act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   # build_norm_layer(norm_layer, channel)[1]
                                   nn.BatchNorm2d(channel),
                                   )

    def forward(self, x):
        out = self.block(x)
        return out


# class Gaussian(nn.Module):
#     def __init__(self, dim, outdim, size, sigma, norm_layer, act_layer, feature_extra=True):
#         super().__init__()
#         self.feature_extra = feature_extra
#         gaussian = self.gaussian_kernel(size, sigma)
#         gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
#         self.gaussian = nn.Conv2d(dim, outdim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
#         self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
#         self.gaussian.weight.requires_grad_(False)
#         # self.norm = build_norm_layer(norm_layer, dim)[1]
#         self.norm = nn.BatchNorm2d(dim)
#         self.act = act_layer()
#         if feature_extra == True:
#             self.conv_extra = Conv_Extra(dim, norm_layer, act_layer)
#
#     def forward(self, x):
#         edges_o = self.gaussian(x)
#         gaussian = self.act(self.norm(edges_o))
#         if self.feature_extra == True:
#             out = self.conv_extra(x + gaussian)
#         else:
#             out = gaussian
#         return out
#
#     def gaussian_kernel(self, size: int, sigma: float):
#         kernel = torch.FloatTensor([
#             [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
#              for x in range(-size // 2 + 1, size // 2 + 1)]
#             for y in range(-size // 2 + 1, size // 2 + 1)
#         ]).unsqueeze(0).unsqueeze(0)
#         return kernel / kernel.sum()


class MultiScaleLoGFilter(nn.Module):
    def __init__(self, in_c, out_c, sigmas, norm_layer, act_layer):
        super(MultiScaleLoGFilter, self).__init__()
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)
        self.log_convs = nn.ModuleList()
        self.sigmas = sigmas

        for sigma in sigmas:
            k = int(2 * round(3 * sigma) + 1)  # kernel size ~ 6σ + 1
            ax = torch.arange(-(k // 2), (k // 2) + 1, dtype=torch.float32)
            xx, yy = torch.meshgrid(ax, ax, indexing="ij")
            kernel = (xx ** 2 + yy ** 2 - 2 * sigma ** 2) / (2 * math.pi * sigma ** 6) * torch.exp(
                -(xx ** 2 + yy ** 2) / (2 * sigma ** 2)
            )
            kernel = kernel - kernel.mean()
            kernel = kernel / kernel.sum()
            kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

            log_conv = nn.Conv2d(out_c, out_c, kernel_size=k, padding=k // 2, groups=out_c, bias=False)
            log_conv.weight.data = kernel.repeat(out_c, 1, 1, 1)
            log_conv.weight.requires_grad_(False)
            self.log_convs.append(log_conv)

        self.fuse = nn.Conv2d(out_c * len(sigmas), out_c, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(out_c)
        self.norm2 = nn.BatchNorm2d(out_c)
        self.act = act_layer()

    def forward(self, x):
        x = self.conv_init(x)  # [B, C, H, W]
        log_feats = [conv(x) for conv in self.log_convs]  # 每个尺度提取 LoG 特征
        log_concat = torch.cat(log_feats, dim=1)  # [B, C*#scale, H, W]
        log_fused = self.fuse(log_concat)  # 融合不同尺度
        log_act = self.act(self.norm1(log_fused))
        # out = self.norm2(x + log_act)
        out = x + log_act
        return out


class LoGFilter(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, sigma, norm_layer, act_layer):
        super(LoGFilter, self).__init__()
        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)
        """创建高斯-拉普拉斯核"""
        # 初始化二维坐标
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax)
        # 计算高斯-拉普拉斯核
        kernel = (xx ** 2 + yy ** 2 - 2 * sigma ** 2) / (2 * math.pi * sigma ** 6) * torch.exp(
            -(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        # 归一化
        kernel = kernel - kernel.mean()
        kernel = kernel / kernel.sum()
        log_kernel = kernel.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度
        self.LoG = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2),
                             groups=out_c, bias=False)
        self.LoG.weight.data = log_kernel.repeat(out_c, 1, 1, 1)
        self.LoG.weight.requires_grad_(False)
        self.act = act_layer()
        # self.norm1 = build_norm_layer(norm_layer, out_c)[1]
        # self.norm2 = build_norm_layer(norm_layer, out_c)[1]
        self.norm1 = nn.BatchNorm2d(out_c)
        self.norm2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv_init(x)
        LoG = self.LoG(x)
        LoG_edge = self.act(self.norm1(LoG))
        x = self.norm2(x + LoG_edge)
        return x


class SobelFilter(nn.Module):
    def __init__(self, in_c, out_c, norm_layer, act_layer):
        super(SobelFilter, self).__init__()
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)

        # Sobel X 和 Y
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32)

        sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)
        sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)

        self.sobel_x = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1,
                                 groups=out_c, bias=False)
        self.sobel_x.weight.data = sobel_x.repeat(out_c, 1, 1, 1)
        self.sobel_x.weight.requires_grad_(False)

        self.sobel_y = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1,
                                 groups=out_c, bias=False)
        self.sobel_y.weight.data = sobel_y.repeat(out_c, 1, 1, 1)
        self.sobel_y.weight.requires_grad_(False)

        self.act = act_layer()
        self.norm1 = nn.BatchNorm2d(out_c)
        self.norm2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv_init(x)
        grad_x = self.sobel_x(x)
        grad_y = self.sobel_y(x)
        edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        edge = self.act(self.norm1(edge))
        x = self.norm2(x + edge)
        return x


class GaussianFilter(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, sigma, norm_layer, act_layer):
        super(GaussianFilter, self).__init__()
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)

        # 构造高斯核
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        gauss_kernel = kernel.unsqueeze(0).unsqueeze(0)

        self.gauss = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, groups=out_c, bias=False)
        self.gauss.weight.data = gauss_kernel.repeat(out_c, 1, 1, 1)
        self.gauss.weight.requires_grad_(False)

        self.act = act_layer()
        self.norm1 = nn.BatchNorm2d(out_c)
        self.norm2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv_init(x)
        gx = self.gauss(x)
        gx = self.act(self.norm1(gx))
        x = self.norm2(x + gx)
        return x


class LaplacianFilter(nn.Module):
    def __init__(self, in_c, out_c, norm_layer, act_layer):
        super(LaplacianFilter, self).__init__()
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)

        # 拉普拉斯核
        kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32)
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        self.lap = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1,
                             groups=out_c, bias=False)
        self.lap.weight.data = kernel.repeat(out_c, 1, 1, 1)
        self.lap.weight.requires_grad_(False)

        self.act = act_layer()
        self.norm1 = nn.BatchNorm2d(out_c)
        self.norm2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv_init(x)
        lapx = self.lap(x)
        lapx = self.act(self.norm1(lapx))
        x = self.norm2(x + lapx)
        return x


class DRFD(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2)
        self.act_c = act_layer()
        # self.norm_c = build_norm_layer(norm_layer, dim * 2)[1]
        self.norm_c = nn.BatchNorm2d(dim * 2)
        # self.max_m = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_m = nn.Identity()
        # self.norm_m = build_norm_layer(norm_layer, dim * 2)[1]
        self.norm_m = nn.BatchNorm2d(dim * 2)
        self.fusion = nn.Conv2d(dim * 4, self.outdim, kernel_size=1, stride=1)
        # gaussian
        # self.gaussian = GaussianFilter(in_c=self.outdim, out_c=self.outdim, kernel_size=7, sigma=1.0, norm_layer='BN',
        #                                act_layer=act_layer)
        self.gaussian = LoGFilter(self.outdim, self.outdim, 7, 1.0, norm_layer, act_layer)
        # self.gaussian = LaplacianFilter(in_c=self.outdim, out_c=self.outdim, norm_layer='BN', act_layer=act_layer)
        # self.gaussian = SobelFilter(in_c=self.outdim, out_c=self.outdim, norm_layer='BN', act_layer=act_layer)

        # self.norm_g = build_norm_layer(norm_layer, self.outdim)[1]
        self.norm_g = nn.BatchNorm2d(self.outdim)

    def forward(self, x):  # x = [B, C, H, W]
        x = self.conv(x)  # x = [B, 2C, H, W]
        gaussian = self.gaussian(x)
        x = self.norm_g(x + gaussian)
        max = self.norm_m(self.max_m(x))  # m = [B, 2C, H/2, W/2]
        conv = self.norm_c(self.act_c(self.conv_c(x)))  # c = [B, 2C, H/2, W/2]
        x = torch.cat([conv, max], dim=1)  # x = [B, 2C+2C, H/2, W/2]  -->  [B, 4C, H/2, W/2]
        x = self.fusion(x)  # x = [B, 4C, H/2, W/2]     -->  [B, 2C, H/2, W/2]

        return x


class Stem(nn.Module):
    def __init__(self, in_chans, stem_dim, act_layer, norm_layer):
        super().__init__()
        out_c14 = int(stem_dim / 2)
        out_c12 = int(stem_dim / 1)
        # original size to 2x downsampling layer
        self.Conv_D = nn.Sequential(
            nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14),
            nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c12),
            # build_norm_layer(norm_layer, out_c12)[1])
            nn.BatchNorm2d(out_c12))
        # 定义LoG滤波器
        self.LoG = MultiScaleLoGFilter(in_c=in_chans, out_c=stem_dim, sigmas=[1.0, 2.0], norm_layer='BN',
                                       act_layer=nn.ReLU)
        # self.LoG = LoGFilter(in_chans, stem_dim, 7, 1.0, norm_layer, act_layer)
        # self.LoG = GaussianFilter(in_c=in_chans, out_c=out_c14, kernel_size=7, sigma=1.0, norm_layer='BN',
        #                           act_layer=act_layer)
        # self.LoG = LaplacianFilter(in_c=in_chans, out_c=out_c14, norm_layer='BN', act_layer=act_layer)
        # self.LoG = SobelFilter(in_c=in_chans, out_c=out_c14, norm_layer='BN', act_layer=act_layer)

        # # gaussian
        # self.gaussian = Gaussian(out_c12, 7, 1.0, norm_layer, act_layer)
        # self.norm = build_norm_layer(norm_layer, out_c12)[1]
        self.norm = nn.BatchNorm2d(out_c12)
        self.drfd = DRFD(out_c12, norm_layer, act_layer)

    def forward(self, x):
        x = self.LoG(x)
        # x = self.Conv_D(x)
        # x = self.norm(x + self.gaussian(x))
        # x = self.drfd(x)

        return x


class VSSM(nn.Module):
    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 2, 9, 2],
            dims=[96, 192, 384, 768],
            depths_decoder=[2, 2, 2, 2],
            dims_decoder=[768, 384, 192, 96],
            win_size=[8, 4, 2, 1],
            decoder_win_size=[1, 2, 4, 8],
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_simple_init=False,
            # =========================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",
            use_checkpoint=False,
            directions=None,
            decoder_directions=None,
            **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        self.dim_decoder = dims_decoder

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        decoder_dpr = [x.item() for x in
                       torch.linspace(0, drop_path_rate, sum(depths_decoder))]  # stochastic depth decay rule
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        if norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]

        if ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]
        self.stem_dim = 36
        self.patch_embed = nn.Sequential(
            nn.Conv2d(self.stem_dim, dims[0], kernel_size=patch_size, stride=patch_size, bias=True),
            Permute(0, 2, 3, 1),
            (norm_layer(dims[0]) if patch_norm else nn.Identity()),
        )

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = PatchMerging2D(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                norm_layer=norm_layer,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                win_size=win_size[i_layer],
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                # =================
                directions=None if directions is None else directions[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
            ))

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers_up.append(self._make_layer_up(
                dim=self.dim_decoder[i_layer],
                drop_path=decoder_dpr[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                upsample=PatchExpand2D(dim=self.dim_decoder[i_layer], dim_scale=2) if (i_layer != 0) else None,
                win_size=decoder_win_size[i_layer],
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                # =================
                directions=None if decoder_directions is None else decoder_directions[sum(depths_decoder[:i_layer]):sum(
                    depths_decoder[:i_layer + 1])]
            ))

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, 2, 1)

        self.patch_embed2 = nn.Sequential(
            nn.Conv2d(self.stem_dim, dims[0], kernel_size=patch_size, stride=patch_size, bias=True),
            Permute(0, 2, 3, 1),
            (norm_layer(dims[0]) if patch_norm else nn.Identity()),
        )

        self.layers2 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = PatchMerging2D(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                norm_layer=norm_layer,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers2.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                win_size=win_size[i_layer],
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                # =================
                directions=None if directions is None else directions[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
            ))

        self.layers_up2 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers_up2.append(self._make_layer_up(
                dim=self.dim_decoder[i_layer],
                drop_path=decoder_dpr[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                upsample=PatchExpand2D(dim=self.dim_decoder[i_layer], dim_scale=2) if (i_layer != 0) else None,
                win_size=decoder_win_size[i_layer],
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                # =================
                directions=None if decoder_directions is None else decoder_directions[sum(depths_decoder[:i_layer]):sum(
                    depths_decoder[:i_layer + 1])]
            ))

        self.final_up2 = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv2 = nn.Conv2d(dims_decoder[-1] // 4, 1, 1)

        dims2 = [64, 128, 256, 256]
        self.cross_attns = nn.ModuleList()
        self.cross_attns2 = nn.ModuleList()
        for i in range(self.num_layers):
            self.cross_attns.append(CrossAttention(dim_q=dims2[i], dim_kv=512))
            self.cross_attns2.append(CrossAttention(dim_q=dims2[i], dim_kv=512))

        self.gate1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * c, c, kernel_size=1),
                nn.Sigmoid()
            ) for c in dims2
        ])
        self.gate2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * c, c, kernel_size=1),
                nn.Sigmoid()
            ) for c in dims2
        ])

        # *******************************************************************
        self.preprocess = Stem(in_chans=4, stem_dim=self.stem_dim, act_layer=nn.SiLU,
                               norm_layer=dict(type='BN', requires_grad=True))
        self.preprocess2 = Stem(in_chans=2, stem_dim=self.stem_dim, act_layer=nn.SiLU,
                                norm_layer=dict(type='BN', requires_grad=True))
        self.decoder = self.make_layer(3, 8, 2)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            win_size=8,
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_simple_init=False,
            # ===========================
            directions=None,
            **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                win_size=win_size,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                use_checkpoint=use_checkpoint,
                directions=directions[d] if directions is not None else None
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),
            downsample=downsample,
        ))

    @staticmethod
    def _make_layer_up(
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            upsample=None,
            win_size=8,
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_simple_init=False,
            # ===========================
            directions=None,
            **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                win_size=win_size,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                use_checkpoint=use_checkpoint,
                directions=directions[d] if directions is not None else None
            ))
        if upsample is None:
            return nn.Sequential(OrderedDict(
                blocks=nn.Sequential(*blocks, )
            ))
        if upsample is not None:
            return nn.Sequential(OrderedDict(
                upsample=upsample,
                blocks=nn.Sequential(*blocks, ),
            ))

    def make_layer(self, inchannel, outchannel, block_num, stride=1):
        layers = [nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU()
        )]

        for i in range(0, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))

        layers.append(nn.Sequential(
            nn.Conv2d(outchannel, 1, 3, 1, 1, bias=False),
        ))

        return nn.Sequential(*layers)

    def forward_features(self, x, y):
        x = self.preprocess(x)
        y = self.preprocess2(y)
        skip_list = []
        x = self.patch_embed(x)

        skip_list2 = []
        y = self.patch_embed2(y)

        for i in range(len(self.layers)):
            skip_list.append(x)
            x = self.layers[i](x)

            skip_list2.append(y)
            y = self.layers2[i](y)

            if i >= 1:
                x = x.permute(0, 3, 1, 2)
                y = y.permute(0, 3, 1, 2)
                concat = torch.cat([x, y], dim=1)
                gate1_out = self.gate1[i](concat)
                x_new = gate1_out * x + (1 - gate1_out) * y

                gate2_out = self.gate2[i](concat)  # [B, C, H, W]
                y_new = gate2_out * y + (1 - gate2_out) * x
                x = x_new.permute(0, 2, 3, 1)
                y = y_new.permute(0, 2, 3, 1)
                # x = x + y
                # y = x

        return x, skip_list, y, skip_list2

    def forward_features_up(self, x, skip_list, y, skip_list2):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-inx])
        for inx, layer_up in enumerate(self.layers_up2):
            if inx == 0:
                y = layer_up(y)
            else:
                y = layer_up(y + skip_list2[-inx])
        return x, y

    def forward_final(self, x, y):
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        x = self.final_conv(x)
        y = self.final_up2(y)
        y = y.permute(0, 3, 1, 2)
        y = self.final_conv2(y)
        return x, y

    def forward(self, x, y):
        # pngFeatures = self.visualSIEncoder(text)[2]
        x, skip_list, y, skip_list2 = self.forward_features(x, y)
        x, y = self.forward_features_up(x, skip_list, y, skip_list2)
        x, y = self.forward_final(x, y)

        return x, y, y


# compatible with openmmlab
class Backbone_LocalVSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer=nn.LayerNorm, **kwargs):
        print(kwargs['directions'])
        super().__init__(**kwargs)

        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="state_dict"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x)  # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x

        return outs


@register_model
def ELViM(*args, in_chans=3, num_classes=1, dims=[32, 64, 128, 256],
          depths=[1, 1, 1, 1], depths_decoder=[1, 1, 1, 1], dims_decoder=[256, 128, 64, 32],
          d_state=16, drop_path_rate=0.2, **kwargs):

    directions = [
        ['h', 'v', 'h_flip', 'v_flip'],
        ['h', 'v', 'h_flip', 'v_flip'],
        ['h', 'v', 'h_flip', 'v_flip'],
        ['h', 'v', 'h_flip', 'v_flip'],
    ]
    decoder_directions = [
        ['h', 'v', 'h_flip', 'v_flip'],
        ['h', 'v', 'h_flip', 'v_flip'],
        ['h', 'v', 'h_flip', 'v_flip'],
        ['h', 'v', 'h_flip', 'v_flip'],
    ]

    return VSSM(in_chans=in_chans, num_classes=num_classes, dims=dims, depths=depths, dims_decoder=dims_decoder,
                depths_decoder=depths_decoder, d_state=d_state, drop_path_rate=drop_path_rate,
                directions=directions, decoder_directions=decoder_directions)


class EL_Mamba(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 depths=[1, 1, 1, 1],
                 depths_decoder=[1, 1, 1, 1],
                 drop_path_rate=0.0,
                 load_ckpt_path=None,
                 ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        self.network = ELViM(input_channels=input_channels, num_classes=num_classes, depths=depths,
                              depths_decoder=depths_decoder, drop_path_rate=drop_path_rate)

    def forward(self, x, y):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits, logits2, logits3 = self.network(x, y)
        return logits, logits2, logits3
