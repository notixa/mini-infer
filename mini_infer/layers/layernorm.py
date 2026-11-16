import einops
import torch
import triton
import triton.language as tl

from mini_infer.config.engine_config import DeviceConfig


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    w_ptr,
    o_ptr,
    stride_xn,
    stride_on,
    eps,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    x_base = x_ptr + pid * stride_xn
    o_base = o_ptr + pid * stride_on
    offsets = tl.arange(0, BLOCK_DIM)
    mask = offsets < HEAD_DIM
    x = tl.load(x_base + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
    x_square = x * x
    x_sum = tl.sum(x_square, -1)
    scale = tl.rsqrt(x_sum / HEAD_DIM + eps)
    x = ((x * scale) * w.to(tl.float32)).to(x_ptr.dtype.element_ty)
    tl.store(o_base + offsets, x, mask=mask)


@triton.jit
def rmsnorm_with_residual_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    o_ptr,
    stride_xn,
    stride_rn,
    stride_on,
    eps,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    x_base = x_ptr + pid * stride_xn
    r_base = r_ptr + pid * stride_rn
    o_base = o_ptr + pid * stride_on
    offsets = tl.arange(0, BLOCK_DIM)
    mask = offsets < HEAD_DIM
    x = tl.load(x_base + offsets, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_base + offsets, mask=mask, other=0.0)
    x = x + r
    h = x.to(r_ptr.dtype.element_ty)
    tl.store(r_base + offsets, h, mask=mask)
    w = tl.load(w_ptr + offsets, other=0.0, mask=mask)
    x_square = x * x
    x_sum = tl.sum(x_square, -1)
    scale = tl.rsqrt(x_sum / HEAD_DIM + eps)
    x = ((x * scale) * w.to(tl.float32)).to(x_ptr.dtype.element_ty)
    tl.store(o_base + offsets, x, mask=mask)


class RMSNorm(torch.nn.Module):

    def __init__(self, head_dim: int, eps: float = 1e-5, dtype=None):
        super().__init__()
        device = DeviceConfig.device_
        gain = torch.ones(head_dim, device=device, dtype=dtype)
        self.head_dim_ = head_dim
        self.eps_ = eps
        self.weight = torch.nn.Parameter(gain)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        assert x.ndim in (2, 3)
        assert x.shape[-1] == self.head_dim_
        assert x.stride(-1) == 1
        if residual is not None:
            assert residual.shape == x.shape
            assert residual.stride(-1) == 1
            assert residual.dtype == x.dtype
            assert residual.is_contiguous()
        N = x.shape[0]
        orig_ndim = x.ndim
        if orig_ndim == 3:
            x = einops.rearrange(x, "N num_heads head_dim -> (N num_heads) head_dim")
            if residual is not None:
                residual = einops.rearrange(
                    residual, "N num_heads head_dim -> (N num_heads) head_dim")
        D, HEAD_DIM = x.shape
        o = torch.empty_like(x, dtype=x.dtype, device=x.device)
        grid = (D, )
        if residual is None:
            rmsnorm_kernel[grid](
                x,
                self.weight,
                o,
                x.stride(0),
                o.stride(0),
                self.eps_,
                HEAD_DIM=HEAD_DIM,
                BLOCK_DIM=triton.next_power_of_2(HEAD_DIM),
            )
            if orig_ndim == 3:
                o = einops.rearrange(o,
                                     "(N num_heads) head_dim -> N num_heads head_dim",
                                     N=N)
            return o
        else:
            rmsnorm_with_residual_kernel[grid](
                x,
                residual,
                self.weight,
                o,
                x.stride(0),
                residual.stride(0),
                o.stride(0),
                self.eps_,
                HEAD_DIM=HEAD_DIM,
                BLOCK_DIM=triton.next_power_of_2(HEAD_DIM),
            )
            if orig_ndim == 3:
                o = einops.rearrange(o,
                                     "(N num_heads) head_dim -> N num_heads head_dim",
                                     N=N)
                residual = einops.rearrange(
                    residual, "(N num_heads) head_dim -> N num_heads head_dim", N=N)
            return o, residual
