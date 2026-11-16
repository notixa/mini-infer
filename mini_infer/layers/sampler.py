import torch


def _sample_tokens(
    logits: torch.Tensor,
    temperatures: torch.Tensor | float,
) -> torch.Tensor:
    scaled_logits = logits.float() / temperatures
    probs = torch.softmax(scaled_logits, dim=-1)
    noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
    return probs.div_(noise).argmax(dim=-1)


class Sampler(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        assert logits.ndim >= 1
        if temperatures is None:
            return logits.argmax(dim=-1)

        if isinstance(temperatures, (int, float)):
            if temperatures <= 0:
                return logits.argmax(dim=-1)
            return _sample_tokens(logits, float(temperatures))

        assert logits.ndim == 2
        assert temperatures.ndim == 1
        assert logits.shape[0] == temperatures.shape[0]

        greedy_tokens = logits.argmax(dim=-1)
        safe_temps = temperatures.float().masked_fill(temperatures <= 0, 1.0)
        sample_tokens = _sample_tokens(logits, safe_temps.unsqueeze(-1))
        return torch.where(temperatures > 0, sample_tokens, greedy_tokens)
