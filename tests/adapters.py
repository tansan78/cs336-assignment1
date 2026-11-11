from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO, Iterator

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

import einops
import regex as re
import logging
import math

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    return einops.einsum(
        in_features, weights,
        " ... d_in, d_out d_in -> ... d_out"
    )


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    return weights[token_ids]


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight

    # FLOPS: batch_size * (2 * seq_length * d_model * d_ff)
    gate = run_silu(
        einops.einsum(in_features, w1_weight, '... d_model, d_ff d_model -> ... d_ff')
    )

    gate_signal = einops.einsum(in_features, w3_weight, '... d_model, d_ff d_model -> ... d_ff')

    activation = gate * gate_signal


    return einops.einsum(
        activation, w2_weight, '... d_ff, d_model d_ff -> ... d_model'
    )


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]
    # FLOPS: batch_size * num_head * (2 * sequence_length * sequence_length * (d_model / num_head))
    qk = einops.einsum(Q, K, '... queries d_k, ... keys d_k -> ... queries keys') / (d_k ** 0.5)
    masked_qk = torch.where(mask, qk, -torch.inf)

    normalized_qk = run_softmax(masked_qk, dim=-1)
    # FLOPS: batch_size * num_head * (2 * sequence_length * sequence_length * (d_model / num_head))
    return einops.einsum(normalized_qk, V, '... queries keys, ... keys d_v -> ... queries d_v')


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    # 
    qkv_weights = torch.stack([q_proj_weight, k_proj_weight, v_proj_weight])
    qkv = einops.einsum(in_features, qkv_weights, 
                        ' ... sequence_length d_model, stack h_dkv d_model -> ... sequence_length stack h_dkv')
    
    # ... num_heads sequence_length  head_dim
    Q = einops.rearrange(qkv[..., 0, :], 
                         '... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k', num_heads=num_heads)
    K = einops.rearrange(qkv[..., 1, :], 
                         '... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k', num_heads=num_heads)
    V = einops.rearrange(qkv[..., 2, :], 
                         '... sequence_length (num_heads d_v) -> ... num_heads sequence_length d_v', num_heads=num_heads)
    
    seq_len = Q.shape[-2]
    mask = torch.triu(
        torch.ones((*Q.shape[:-1], seq_len), dtype=torch.bool, device=in_features.device))\
            .transpose(-1, -2)

    # ... num_heads sequence_length d_v
    sdpa = run_scaled_dot_product_attention(Q, K, V, mask)

    # ...  sequence_length num_heads d_v
    mha = einops.rearrange(sdpa, '... num_heads sequence_length d_v -> ... sequence_length (num_heads d_v)', num_heads=num_heads)

    return einops.einsum(
        o_proj_weight, mha, 'd_model h_dv, ... sequence_length h_dv -> ... sequence_length d_model'
    )


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    
    qkv_weights = torch.stack([q_proj_weight, k_proj_weight, v_proj_weight])
    # FLOPS: batch_size * 3 * (2 * sequence_length * d_model * d_model)
    qkv = einops.einsum(in_features, qkv_weights, 
                        ' ... sequence_length d_model, stack h_dkv d_model -> ... sequence_length stack h_dkv')
    
    # ... num_heads sequence_length  head_dim
    Q = einops.rearrange(qkv[..., 0, :], 
                         '... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k', num_heads=num_heads)
    K = einops.rearrange(qkv[..., 1, :], 
                         '... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k', num_heads=num_heads)
    V = einops.rearrange(qkv[..., 2, :], 
                         '... sequence_length (num_heads d_v) -> ... num_heads sequence_length d_v', num_heads=num_heads)
    
    seq_len = Q.shape[-2]
    # can I save memory by use simple size 1 for all dimensions except the last one
    mask = torch.triu(
        torch.ones((*Q.shape[:-1], seq_len), dtype=torch.bool, device=in_features.device))\
            .transpose(-1, -2)

    # ... num_heads sequence_length d_v
    d_k = Q.shape[-1]
    Q_with_rope = run_rope(d_k, theta, max_seq_len, Q, token_positions)
    K_with_rope = run_rope(d_k, theta, max_seq_len, K, token_positions)
    sdpa = run_scaled_dot_product_attention(Q_with_rope, K_with_rope, V, mask)

    # ...  sequence_length num_heads d_v
    mha = einops.rearrange(sdpa, '... num_heads sequence_length d_v -> ... sequence_length (num_heads d_v)', num_heads=num_heads)

    # FLOPS: batch_size * (2 * sequence_length * d_model * d_model)
    return einops.einsum(
        o_proj_weight, mha, 'd_model h_dv, ... sequence_length h_dv -> ... sequence_length d_model'
    )


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    device = in_query_or_key.device

    base = 1 / (theta ** (torch.arange(start=0, end=d_k, step=2, dtype=torch.float) / d_k))
    # (max_seq_len, d_k // 2)
    theta_matrix = torch.outer(
        torch.arange(start=0, end=max_seq_len, step=1, dtype=torch.float),
        base
    ).to(device)
    cos = torch.cos(theta_matrix)
    sin = torch.sin(theta_matrix)
    # print(f"{base.dtype}, {theta_matrix.dtype}, {cos.dtype}")

    # separate even and odd elements
    even_in_query_or_key = in_query_or_key[..., 0::2]
    odd_in_query_or_key = in_query_or_key[..., 1::2]

    even_result = even_in_query_or_key * cos[token_positions, :] \
        - odd_in_query_or_key * sin[token_positions, :]
    odd_result = even_in_query_or_key * sin[token_positions, :] \
        + odd_in_query_or_key * cos[token_positions, :]
    
    return torch.stack((even_result, odd_result), dim=-1).flatten(start_dim=-2)



def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    eps = 1e-5

    rmsnorm = run_rmsnorm(d_model, eps, weights['ln1.weight'], in_features)

    batch_size, sequence_length, _ = in_features.shape
    token_positions = torch.arange(start=0, end=sequence_length, step=1, dtype=torch.int).unsqueeze(0)
    mha = run_multihead_self_attention_with_rope(d_model, num_heads, max_seq_len, theta,
                                           weights['attn.q_proj.weight'],
                                           weights['attn.k_proj.weight'],
                                           weights['attn.v_proj.weight'],
                                           weights['attn.output_proj.weight'],
                                           rmsnorm, 
                                           token_positions)
    
    combined = in_features + mha

    rmsnorm2 = run_rmsnorm(d_model, eps, weights['ln2.weight'], combined)

    return combined + run_swiglu(d_model, d_ff, weights['ffn.w1.weight'], weights['ffn.w2.weight'], weights['ffn.w3.weight'], rmsnorm2)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    
    embs_tbl = weights['token_embeddings.weight']
    # print(in_indices.dtype, in_indices.shape)
    embs = embs_tbl[in_indices.to(torch.int)]


    for layer_idx in range(num_layers):
        layer_weights = {
            'attn.q_proj.weight': weights[f'layers.{layer_idx}.attn.q_proj.weight'],
            'attn.k_proj.weight': weights[f'layers.{layer_idx}.attn.k_proj.weight'],
            'attn.v_proj.weight': weights[f'layers.{layer_idx}.attn.v_proj.weight'],
            'attn.output_proj.weight': weights[f'layers.{layer_idx}.attn.output_proj.weight'],
            'ln1.weight': weights[f'layers.{layer_idx}.ln1.weight'],
            'ffn.w1.weight': weights[f'layers.{layer_idx}.ffn.w1.weight'],
            'ffn.w2.weight': weights[f'layers.{layer_idx}.ffn.w2.weight'],
            'ffn.w3.weight': weights[f'layers.{layer_idx}.ffn.w3.weight'],
            'ln2.weight': weights[f'layers.{layer_idx}.ln2.weight'],
        }
        embs = run_transformer_block(d_model, num_heads, d_ff, context_length, rope_theta, layer_weights, embs)
    

    eps = 1e-5
    embs = run_rmsnorm(d_model, eps, weights['ln_final.weight'], embs)

    # FLOPS: batch_size * (2 * sequence_length * d_model * vocab_size)
    return einops.einsum(embs, weights['lm_head.weight'], 
                         'batch_size sequence_length d_model, vocab_size d_model -> batch_size sequence_length vocab_size')


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    in_dtype = in_features.dtype

    in_features = in_features.to(torch.float32)
    rms_a = torch.sqrt(
        einops.reduce(
            einops.einsum(in_features, in_features, '... d_model, ... d_model -> ... d_model'),
            '... d_model -> ... 1', 'sum'
        ) / d_model + eps
    )

    rms_norm = einops.einsum(in_features, weights, '... d_model, d_model -> ... d_model') \
        / rms_a

    return rms_norm.to(in_dtype)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return in_features / (1 + torch.exp(-1 * in_features))


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    data_len = dataset.shape[-1]
    starting_index = torch.randint(high=(data_len - context_length), size=(batch_size, 1))
    expanded_index = starting_index.expand(-1, context_length)

    offsets = torch.arange(0, context_length, 1).unsqueeze(0)
    batch_index = expanded_index + offsets
    batch = dataset[batch_index]

    label_offsets = torch.arange(1, context_length + 1, 1).unsqueeze(0)
    label_index = expanded_index + label_offsets
    labels = dataset[label_index]

    return torch.from_numpy(batch).to(device), torch.from_numpy(labels).to(device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    leveled_in_features = in_features - torch.max(in_features, dim=dim, keepdim=True)[0]
    return torch.exp(leveled_in_features) / torch.sum(torch.exp(leveled_in_features), dim=dim, keepdim=True)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    leveled_input = inputs - torch.max(inputs, dim=-1, keepdim=True)[0]

    # print(inputs.shape, targets.shape)
    batch_indices = torch.arange(inputs.shape[0], device=inputs.device)
    nominator_val = torch.sum(leveled_input[batch_indices, targets.to(torch.int)])
    
    target_logits = leveled_input[batch_indices, targets.to(torch.int)]
    nominator_val = torch.sum(target_logits)

    demonitor_offet = torch.sum(
        torch.log(
            torch.sum(
                torch.exp(leveled_input), 
                dim=-1)))

    elem = leveled_input.numel() / leveled_input.shape[-1]
    
    return -1 * (nominator_val - demonitor_offet) / (elem)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    eps = 1e-6

    grads = [p.grad for p in parameters if p.grad is not None]

    l2_norm = torch.sqrt(
        sum(
            torch.sum(g.pow(2)) for g in grads))
    if l2_norm < max_l2_norm:
        return
    
    c = max_l2_norm / (l2_norm + eps)
    for g in grads:
        with torch.no_grad():
            g.mul_(c)


class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(self, params, betas, weight_decay, lr=1e-3, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, 'eps': eps, 'betas': betas, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for index, group in enumerate(self.param_groups):
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group["params"]:
                if p.grad is None:
                    continue

                p.data -= lr * weight_decay * p.data

                state = self.state[p]
                if 'm' not in state:
                    state['m'] = torch.zeros(p.shape, device=p.device)
                if 'v' not in state:
                    state['v'] = torch.zeros(p.shape, device=p.device)

                grad = p.grad.data
                # state['m'] = beta1 * state['m'] + (1 - beta1) * grad
                # state['v'] = beta2 * state['v'] + (1 - beta2) * grad * grad
                state['m'].mul_(beta1).add_(grad, alpha=1 - beta1)
                state['v'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                t = state.get("t", 1)
                state["t"] = t + 1

                lr_t = lr * (math.sqrt(1 - beta2 ** t)) / (1 - beta1 ** t)
                p.data -= lr_t * state['m'] / (torch.sqrt(state['v']) + eps)

        return loss
    

def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamWOptimizer


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    output_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(output_dict, out)
    return out


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    output_dict = torch.load(src)
    if model:
        model.load_state_dict(output_dict["model"])
    if optimizer:
        optimizer.load_state_dict(output_dict["optimizer"])
    return output_dict["iteration"]


class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        self._vocab = vocab
        self._merges = merges
        self._special_tokens = special_tokens

        # logging.info(f"{self._vocab[0]}, {self._vocab[1]}, {self._vocab[2]}")
        if self._special_tokens:
            sorted_special_tokens = sorted(special_tokens, key=lambda x: len(x), reverse=True)
            escaped_special_tokens = ['(' + re.escape(s) + ')' for s in sorted_special_tokens]
            self._special_token_pat = "|".join(escaped_special_tokens)
        else:
            self._special_token_pat = None

        self._bytes_to_id = {bs : id for id, bs in vocab.items()}
        
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    def get_voc_size(self):
        return max(self._vocab.keys()) + 1
    
    def encode(self, text):
        results = []
        if self._special_token_pat:
            split_texts = re.split(self._special_token_pat, text)
        else:
            split_texts = [text]

        for split_text in split_texts:
            if split_text == None or split_text == '':
                continue
            
            if self._special_tokens and split_text in self._special_tokens:
                results.append(
                    self._bytes_to_id[split_text.encode("utf-8")])
            else:
                results.extend(self.encode_splitted_text(split_text))
        
        return results

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for item in iterable:
            results = self.encode(item)

            yield from results
    
    def encode_splitted_text(self, text):
        result = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for match in re.finditer(PAT, text):
            match_bytes = match.group().encode("utf-8")
            result.extend(self.encode_one_token(match_bytes))
        
        return result

    def encode_one_token(self, token):
        token_bytes = [bytes([b]) for b in token]
        for m in self._merges:
            new_token_bytes = []
            i = 0

            total_byts = len(token_bytes)
            matched = False
            while i < total_byts:
                if token_bytes[i] == m[0] \
                        and i < total_byts - 1 and token_bytes[i+1] == m[1]:
                    if not matched:
                        new_token_bytes.extend(token_bytes[:i])
                        matched = True
                    new_token_bytes.append(token_bytes[i] + token_bytes[i+1])
                    i += 2
                else:
                    if matched:
                        new_token_bytes.append(token_bytes[i])
                    i += 1
        
            if matched:
                token_bytes = new_token_bytes
        
        return [self._bytes_to_id[bs] for bs in token_bytes]

    def decode(self, ids):
        result = []
        for id in ids:
            result.append(self._vocab.get(id, "\ufffd".encode("utf-8")))

        return b"".join(result).decode("utf-8")


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    from pathlib import Path
    import regex as re
    import logging

    logging.info('Start ...')
    vocab = {}
    index = 0
    for s in special_tokens:
        vocab[index] = s.encode('utf-8')
        index += 1
    for i in range(256):
        vocab[index] = i.to_bytes(1)
        index += 1

    logging.info('Start loading file')
    contents = Path(input_path).read_text(encoding="utf-8")

    escaped_special_tokens = [re.escape(s) for s in special_tokens]
    special_token_pattern = "|".join(escaped_special_tokens)

    logging.info('pretokenizing...')
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    word_freq = {}
    for content in re.split(special_token_pattern, contents):
        for match in re.finditer(PAT, content):
            match_bytes = match.group().encode("utf-8")
            btuple = tuple([bytes([b]) for b in match_bytes])
            if btuple not in word_freq:
                word_freq[btuple] = 0
            word_freq[btuple] += 1
    del contents

    logging.info('create vocabulary...')
    merges = [] 
    while index < vocab_size:
        if index % 100 == 0:
            logging.info(f"loop {index}")
        voc_freq = get_voc_freq(word_freq)
        max_bi_bytes, _ = get_max_freq(voc_freq)

        vocab[index] = max_bi_bytes[0] + max_bi_bytes[1]
        merges.append(max_bi_bytes)
        index += 1

        word_freq = update_word_freq(word_freq, max_bi_bytes)
        

    return vocab, merges


def get_voc_freq(word_freq):
    voc_freq = {}
    for bytes_tuple, cnt in word_freq.items():
        for i in range(len(bytes_tuple) - 1):
            bi_bytes = (bytes_tuple[i], bytes_tuple[i+1])
            if bi_bytes not in voc_freq:
                voc_freq[bi_bytes] = 0
            
            voc_freq[bi_bytes] += cnt
    
    return voc_freq


def get_max_freq(voc_freq):
    max_bi_bytes = None
    max_cnt = 0

    for bi_bytes, cnt in voc_freq.items():
        if cnt > max_cnt:
            max_bi_bytes = bi_bytes
            max_cnt = cnt
        elif cnt == max_cnt and bi_bytes > max_bi_bytes:
            max_bi_bytes = bi_bytes
    
    return max_bi_bytes, max_cnt


def update_word_freq(word_freq, max_bi_bytes):
    new_word_freq = {}
    for bytes_tuple, cnt in word_freq.items():

        new_bytes_list = []
        index = 0
        while index < len(bytes_tuple):
            if index < len(bytes_tuple) - 1 and \
                 ((bytes_tuple[index], bytes_tuple[index+1]) == max_bi_bytes):
                new_bytes_list.append(bytes_tuple[index] +  bytes_tuple[index+1])
                index += 2
            else:
                new_bytes_list.append(bytes_tuple[index])
                index += 1
        
        new_word_freq[tuple(new_bytes_list)] = cnt
    
    return new_word_freq

