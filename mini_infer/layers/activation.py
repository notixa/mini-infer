import torch
import torch.nn.functional as F


class SiluAndMul(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 1
        assert x.shape[-1] % 2 == 0
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up
