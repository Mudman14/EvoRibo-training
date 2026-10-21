"""Configuration for the EvoRibo pair encoder."""

from dataclasses import dataclass


@dataclass
class SSPairEncoderConfig:
    """Hyperparameters for an ESMFold-style, BPP-aware pair encoder."""

    resm_single_dim: int = 640
    resm_pair_dim: int = 600

    d_s: int = 256
    d_z: int = 128
    num_blocks: int = 8
    num_recycles: int = 1

    sequence_head_width: int = 32
    pairwise_head_width: int = 32
    position_bins: int = 32
    dropout: float = 0.1

    vocab_size: int = 6
    pad_idx: int = 5

    use_bpp_level1: bool = True
    use_bpp_level2_bias: bool = True
    use_bpp_level3_gate: bool = True

    bpp_eps: float = 1e-4
    ss_bias_hidden: int = 64
    ss_gate_hidden: int = 64
    ss_bias_scale_init: float = -4.0
    ss_gate_scale_init: float = -4.0
    ss_gate_amplitude: float = 0.5

    return_pair_sym: bool = False

    use_gradient_checkpointing: bool = False
    triangle_attention_chunk_size: int = 0
