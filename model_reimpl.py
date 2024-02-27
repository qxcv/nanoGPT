"""
GPT2-style Transformer model. Some things I changed:

* No biases in layer norm nor in most linear layers.
* Uses sinusoidal positional embeddings, not learned ones.
"""
from typing import Any, Optional

import einops as eo
import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn
import inspect

from model import GPTConfig

"""
TODO(sam): implement (k,v) cache

Design decisions I need to make about k/v caching:

- do I cache locally inside the attention layer, or do I have higher-level code
  thread the k/v cache through function calls?
- how do I disable caching at train time, and reset the cache for different
  seqs?
- how do I keep start positions correct for input sequences?

I suspect the threading design is probably easiest, since I can infer the start
position for sequences from the length of the k/v cache, and the caller has full
control over whether they want to use the k/v cache at all.
"""


def rsqrt(x: float) -> float:
    """1/sqrt(x)"""
    return th.rsqrt(th.as_tensor(x)).item()


class Dropout(nn.Module):
    """Dropout, drops (p*100)% of inputs during training."""

    def __init__(self, *, p: float) -> None:
        super().__init__()
        # can't drop 100%, can drop 0%
        assert 0 <= p < 1, f"{p=} not in [0,1)"
        self.p = p

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.training and self.p > 0:
            # drop (100*p)% of inputs
            # (but rescale to maintain expected value)
            keep_mask = (th.rand_like(x) > self.p).to(x.dtype)
            x = x * keep_mask / (1 - self.p)
        # if it's not training mode, then dropout is a noop
        return x


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal absolute positional embeddings."""

    def forward(self, x: th.Tensor) -> None:
        # TODO(sam): cache this (low priority)
        with th.no_grad():
            # shape [b, t, d]
            _, t, d = x.shape
            assert d % 2 == 0, "{d=}, but must be even so we can alternate sin & cos"
            halfdim_arange = th.arange(d // 2, device=x.device, dtype=x.dtype)
            time_arange = th.arange(t, device=x.device, dtype=x.dtype)
            scale_factor = th.pow(1 / 10000, (2 / d) * halfdim_arange)
            # outer product to get sin & cos arguments
            input_mat = time_arange[:, None] * scale_factor[None, :]
            sines_cosines = th.stack((th.sin(input_mat), th.cos(input_mat)), dim=-1)
            # interleave sines and cosines, and also add a dummy leading dim for
            # broadcast addition
            interleaved_pos = eo.rearrange(
                sines_cosines,
                "t half_d two -> t (half_d two)",
                half_d=d // 2,
                two=2,
            )[None]
            assert (
                interleaved_pos.shape[1:] == x.shape[1:]
            ), f"{interleaved_pos.shape=} does not match {x.shape=} at positions 1-end"

        return x + interleaved_pos


class EmbeddingUnembeddingLayer(nn.Module):
    """Embedding and un-embedding tokens"""

    w: nn.Parameter

    def __init__(
        self,
        *,
        d: int,
        vocab_size: int,
        device: Optional[th.DeviceObjType] = None,
        dtype: Optional[th.dtype] = None,
    ) -> None:
        super().__init__()
        kwargs = dict(device=device, dtype=dtype)
        self.register_parameter("w", nn.Parameter(th.empty((vocab_size, d), **kwargs)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with th.no_grad():
            self.w.normal_(std=0.02)

    def embed(self, x: th.Tensor) -> th.Tensor:
        """Embed tokens."""
        assert x.dtype == th.long, f"dtype must be long, but {x.dtype=}"
        return self.w[x.long()]

    def logits(self, x: th.Tensor) -> th.Tensor:
        return th.einsum("btd,vd->btv", x, self.w)

    def forward(self, x: th.Tensor) -> None:
        raise NotImplementedError(
            "Don't call forward(). Instead, use .embed() and .logits()"
        )


class Linear(nn.Module):
    """Re-implementation of dense linear layer."""

    w: nn.Parameter
    b: Optional[nn.Parameter]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        device: Optional[th.DeviceObjType] = None,
        dtype: Optional[th.dtype] = None,
    ) -> None:
        super().__init__()
        kwargs = dict(device=device, dtype=dtype)
        self.register_parameter(
            "w", nn.Parameter(th.empty((in_dim, out_dim), **kwargs))
        )
        self.b = None
        if bias:
            self.register_parameter("b", nn.Parameter(th.empty((out_dim,), **kwargs)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with th.no_grad():
            self.w.normal_(std=0.02)
            if self.b is not None:
                self.b.zero_()

    def forward(self, x: th.Tensor) -> th.Tensor:
        # batch matmul
        out = th.einsum("...ij,...jk->...ik", x, self.w)
        if self.b is not None:
            # self.b should broadcast out
            out = out + self.b
        return out


class LayerNorm(nn.Module):
    """Layer norm, applied along last axis."""

    scale: nn.Parameter
    bias: Optional[nn.Parameter]

    def __init__(
        self,
        *,
        d: int,
        bias: bool = False,
        eps: float = 1e-5,
        device: Optional[th.DeviceObjType] = None,
        dtype: Optional[th.dtype] = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        kwargs = dict(device=device, dtype=dtype)
        self.register_parameter("scale", nn.Parameter(th.empty((d,), **kwargs)))
        if bias:
            self.register_parameter("bias", nn.Parameter(th.empty((d,))))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with th.no_grad():
            self.scale.normal_(std=0.02)
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, x: th.Tensor) -> th.Tensor:
        std = x.std(dim=-1, keepdim=True).clamp(min=self.eps)
        mean = x.std(dim=-1, keepdim=True)
        x = self.scale * (x - mean) / std
        if self.bias is not None:
            x = x + self.bias
        return x


class CausalAttention(nn.Module):
    """Causally-masked self-attention."""

    def __init__(
        self,
        *,
        d: int,
        n_heads: int,
        device: Optional[th.DeviceObjType] = None,
        dtype: Optional[th.dtype] = None,
    ) -> None:
        super().__init__()
        kwargs = dict(device=device, dtype=dtype)
        self.n_heads = n_heads
        self.model_dim = d
        # 1/sqrt(d) scale factor
        self.scale_factor = rsqrt(d)
        assert d % n_heads == 0, f"{n_heads=} not divisible by {d=}"
        self.wq = Linear(d, d, bias=False, **kwargs)
        self.wk = Linear(d, d, bias=False, **kwargs)
        self.wv = Linear(d, d, bias=False, **kwargs)
        # for some reason gpt2 has a second projection
        self.proj = Linear(d, d, bias=False, **kwargs)

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, t, d = x.shape

        def split_heads(combined: th.Tensor) -> th.Tensor:
            return eo.rearrange(
                combined,
                "B T (H Dp) -> B T H Dp",
                B=b,
                T=t,
                H=self.n_heads,
                Dp=self.model_dim // self.n_heads,
            )

        def combine_heads(split: th.Tensor) -> th.Tensor:
            return eo.rearrange(
                split,
                "B T H Dp -> B T (H Dp)",
                B=b,
                T=t,
                H=self.n_heads,
                Dp=self.model_dim // self.n_heads,
            )

        # TODO(sam): k/v cache
        # (not sure of the best way to architect the k/v cache; whatever I do
        # has to play nice with positional embeddings, and would ideally avoid
        # running MLPs over the cached positions as well)
        q = split_heads(self.wq(x))
        k = split_heads(self.wk(x))
        v = split_heads(self.wv(x))

        # compute T*T matrix of inner products
        # (q is the time axis for queries, k is the time axis for keys)
        # (this moves the h axis one place to the left so that the next einsum
        # is a bit more efficient)
        logits = th.einsum("bqhd,bkhd -> bhqk", q, k) * self.scale_factor

        # causal masking
        causal_mask = th.full_like(logits[0, 0], fill_value=-th.inf, requires_grad=False)
        # diagonal=1 means zero out everything *at or below* the main
        # diagonal (everything else will be -inf)
        causal_mask.triu_(diagonal=1)

        # sum out the k axis (time dimension corresponding to keys)
        attn_weights = th.softmax(logits + causal_mask[None, None], dim=-1)
        attn_out = th.einsum("bhqk,bkhv -> bqhv", attn_weights, v)
        combined = combine_heads(attn_out)
        projected = self.proj(combined)

        expected_shape = (b, t, d)
        assert (
            projected.shape == expected_shape
        ), f"{projected.shape=} != {expected_shape}"

        return projected


class MLP(nn.Module):
    def __init__(
        self,
        *,
        d: int,
        device: Optional[th.DeviceObjType] = None,
        dtype: Optional[th.dtype] = None,
    ) -> None:
        super().__init__()
        kwargs = dict(device=device, dtype=dtype)
        # these do have biases? IDK.
        self.enc = Linear(d, 4 * d, **kwargs)
        # also we give 'proj' the same name as the linear projection in the
        # attention layer, so that we can scale the weight init down by
        # 1/(sqrt(2*L)) later on
        self.proj = Linear(4 * d, d, **kwargs)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.proj(F.gelu(self.enc(x)))


class GPTBlock(nn.Module):
    def __init__(self, *, d: int, n_heads: int, p_drop: float) -> None:
        super().__init__()
        self.ln1 = LayerNorm(d=d)
        self.mlp = MLP(d=d)
        self.ln2 = LayerNorm(d=d)
        self.attn = CausalAttention(d=d, n_heads=n_heads)
        self.drop = Dropout(p=p_drop)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # TODO(sam): K/V cache
        # attn_in = self.ln1(x)
        # attn_result = self.attn(attn_in)
        # x = x + self.drop(attn_result)
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class GPTReimplementation(nn.Module):
    """A very minimal implementation of GPT2.
    (with some small changes, like sinusoidal embeddings)"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        d: int = config.n_embd
        n_layers: int = config.n_layer
        vocab_size: int = config.vocab_size
        n_heads: int = config.n_head
        p_drop: float = config.dropout

        self.drop = Dropout(p=p_drop)
        self.embed_layer = EmbeddingUnembeddingLayer(d=d, vocab_size=vocab_size)
        # TODO(sam): use a learned positional embedding instead of sinusoidal
        # embedding
        self.positional_embedding = SinusoidalEmbedding()
        self.blocks = nn.ModuleList(
            GPTBlock(d=d, n_heads=n_heads, p_drop=p_drop) for _ in range(n_layers)
        )
        self.final_ln = LayerNorm(d=d)

        # extra init logic: divide init for things that project into the
        # residual stream by 1/sqrt(2*L) (there are 2L such things)
        with th.no_grad():
            n_changed = 0
            scale_factor = rsqrt(2 * n_layers)
            for param_name, param in self.named_parameters():
                if param_name.endswith(".proj.w"):
                    param[:] *= scale_factor
                    n_changed += 1
            assert (
                n_changed == 2 * n_layers
            ), f"{n_changed=}, but expected to change{2*n_layers=} parameter tensors"

        print(f"number of parameters: {self.get_num_params()/1e6:.2f}M")

    def forward(
        self, inputs: th.Tensor, targets: Optional[th.Tensor] = None
    ) -> tuple[th.Tensor, Optional[th.Tensor]]:
        x = self.embed_layer.embed(inputs)
        x = self.drop(x + self.positional_embedding(x))
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        logits = self.embed_layer.logits(x)
        loss = None
        if targets is not None:
            # target shape should be [b,t]
            loss = F.cross_entropy(
                logits.view((-1, logits.shape[-1])),
                targets.view((-1,)),
                reduction="mean",
                ignore_index=-1,
            )
        return logits, loss

    ######## EVERYTHING BELOW HERE IS COPIED WITH SLIGHT CHANGES ########

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        # assert block_size <= self.config.block_size
        # self.config.block_size = block_size
        # self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        # for block in self.transformer.h:
        #     if hasattr(block.attn, 'bias'):
        #         block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
        pass  # not needed, we're using sinusoids

    # copied from original model.py file
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= sum(p.numel() for p in self.embed_layer.parameters())
        return n_params

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        raise NotImplementedError("haven't implemented GPT2 loading logic")

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(th.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = th.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @th.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = th.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = th.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = th.cat((idx, idx_next), dim=1)
