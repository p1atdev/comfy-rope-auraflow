import torch
import torch.nn as nn


def get_image_position_indices(height: int, width: int):
    pos_idx = torch.zeros(height // 2, width // 2, 3)
    # we're going to make a repeat of (zero, y, x) format

    # y position [[0], [1], [2], ... [height // 2]]
    pos_idx[..., 1] = pos_idx[..., 1] + torch.arange(height // 2).unsqueeze(-1)

    # x position [[0, 1, 2, ... width // 2]]
    pos_idx[..., 2] = pos_idx[..., 2] + torch.arange(width // 2).unsqueeze(0)

    # flatten
    pos_idx = pos_idx.reshape(-1, 3)

    # (height // 2) x (width // 2) x 3 size tensor
    # (zero, y, x) format
    # [[0, 0, 0], [0, 0, 1], [0, 0, 2], ... [0, 0, width // 2],
    #  [0, 1, 0], [0, 1, 1], [0, 1, 2], ... [0, 1, width // 2],
    #  ...
    #  [0, height // 2, 0], [0, height // 2, 1], [0, height // 2, 2], ... [0, height // 2, width // 2]]

    return pos_idx


def get_text_position_indices(
    seq_len: int,
    dim_sizes: list[int],
) -> torch.Tensor:
    return torch.zeros(
        seq_len,
        len(dim_sizes),
    )


# ref: https://github.com/black-forest-labs/flux/blob/main/src/flux/math.py
def _get_rope_frequencies(
    position_indices: torch.Tensor,  # (height//2 * width//2,)
    dim: int,  # positional encoding dimension (16 or 56)
    theta: int,  # the rope theta (normally 10000)
) -> torch.Tensor:
    assert dim % 2 == 0, "dim must be even"

    # 0~1 with dim//2 steps
    scale = (
        torch.arange(0, dim, 2, dtype=torch.float64, device=position_indices.device)
        / dim
    )

    omega = 1.0 / (theta**scale)

    # (height//2 * width//2, 3, dim//2)
    angles = torch.outer(position_indices, omega)

    cos = torch.cos(angles)  # (height//2 * width//2, dim//2)
    sin = torch.sin(angles)

    frequencies = torch.stack([cos, sin], dim=-1)

    return frequencies.float()


def get_rope_frequencies(
    position_indices: torch.Tensor,  # (height//2 * width//2, n_axes)
    dim_sizes: list[int],  # positional encoding dimension ([16, 56, 56])
    theta: int,  # the rope theta (normally 10000)
) -> torch.Tensor:
    assert (
        len(dim_sizes) == position_indices.shape[-1]
    ), "dim_sizes must have the same length as position_indices.shape[-1]"

    # get each axes frequencies
    freqs = torch.cat(
        [
            _get_rope_frequencies(position_indices[..., i], dim, theta)
            for i, dim in enumerate(dim_sizes)
        ],
        dim=-2,
    )

    return freqs


# ref: https://github.com/huggingface/diffusers/blob/fdcbbdf0bb4fb6ae3c2b676af525fced84aa9850/src/diffusers/models/attention_processor.py#L1018-L1025
def applye_rope_frequencies(
    inputs: torch.Tensor,  # (batch_size, seq_len, num_heads, dim)
    freqs: torch.Tensor,
) -> torch.Tensor:
    initial_dtype = inputs.dtype
    # swap seq_len and num_heads dim, and cast to float
    inputs = inputs.transpose(1, 2).float()
    initial_shape = inputs.shape
    freqs = freqs.to(inputs.device)  # move to the same device

    # get cos and sin
    cos, sin = freqs.chunk(2, dim=-1)
    cos, sin = cos.squeeze(-1), sin.squeeze(-1)

    rotated_inputs = (
        # ↓ (batch_size, num_heads, seq_len, dim//2, 2)
        torch.stack(
            [
                # 0::2 to get even index, 1::2 to get odd index
                inputs[..., 0::2] * cos - inputs[..., 1::2] * sin,
                inputs[..., 0::2] * sin + inputs[..., 1::2] * cos,
            ],
            dim=-1,
        ).reshape(*initial_shape)  # reshape to (batch_size, num_heads, seq_len, dim)
    )

    # transpose back
    rotated_inputs = rotated_inputs.transpose(1, 2).to(initial_dtype)

    return rotated_inputs
