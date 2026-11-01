# SPDX-License-Identifier: MIT
# SmolLM2 (Llama) decode step built from TPU-friendly ops, for export to
# ONNX one layer at a time. One new token per call; static KV cache of S slots.
import json
import math

import torch
from safetensors.torch import load_file
from torch import nn

NEG = -10000.0  # additive mask for empty cache slots


class Config:
    def __init__(self, path):
        c = json.load(open(f"{path}/config.json"))
        self.d = c["hidden_size"]
        self.f = c["intermediate_size"]
        self.nh = c["num_attention_heads"]
        self.nkv = c["num_key_value_heads"]
        self.hd = self.d // self.nh
        self.layers = c["num_hidden_layers"]
        self.eps = c["rms_norm_eps"]
        self.theta = c["rope_theta"]
        self.vocab = c["vocab_size"]


def load_weights(path):
    return {k: v.float() for k, v in load_file(f"{path}/model.safetensors").items()}


def rope_tables(cfg, positions):
    """cos, sin of shape [len(positions), hd] (HF non-interleaved layout)."""
    inv = 1.0 / cfg.theta ** (torch.arange(0, cfg.hd, 2).float() / cfg.hd)
    ang = torch.outer(torch.tensor(positions, dtype=torch.float32), inv)
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos(), ang.sin()


def rotate_half_matrix(hd):
    """R with x @ R == cat(-x[hd/2:], x[:hd/2])."""
    h = hd // 2
    r = torch.zeros(hd, hd)
    r[h:, :h] = -torch.eye(h)
    r[:h, h:] = torch.eye(h)
    return r


class RMSNorm(nn.Module):
    # TPU-MLIR fuses the usual pow/mean/rsqrt form into top.RMSNorm, which
    # has no cv18xx lowering. The mean as a MatMul with a 1/d vector avoids it.
    def __init__(self, w, eps):
        super().__init__()
        self.w = nn.Parameter(w, requires_grad=False)
        self.avg = nn.Parameter(torch.full((w.numel(), 1), 1.0 / w.numel()), requires_grad=False)
        self.eps = eps

    def forward(self, x):
        r = torch.reciprocal(torch.sqrt((x * x) @ self.avg + self.eps))
        return x * r * self.w


class DecodeLayer(nn.Module):
    """Inputs: h [1,1,d], cos/sin [1,1,1,hd], k_cache/v_cache [1,nkv,S,hd],
    mask [1,nkv,nh/nkv,S+1]. Outputs: h_out [1,1,d], k_new/v_new [1,nkv,1,hd].
    With delta=True the first output is dh = h_out - h: the caller keeps the
    residual stream (BF16 rounding of the large residual hides a layer's change)."""

    def __init__(self, cfg, w, i, delta=False):
        super().__init__()
        self.delta = delta
        p = f"model.layers.{i}."
        self.cfg = cfg
        self.norm1 = RMSNorm(w[p + "input_layernorm.weight"], cfg.eps)
        self.norm2 = RMSNorm(w[p + "post_attention_layernorm.weight"], cfg.eps)
        lin = lambda n: nn.Parameter(w[p + n + ".weight"].t().contiguous(), requires_grad=False)
        # 1/sqrt(hd) folded into Wq
        self.wq = nn.Parameter(w[p + "self_attn.q_proj.weight"].t() / math.sqrt(cfg.hd), requires_grad=False)
        self.wk = lin("self_attn.k_proj")
        self.wv = lin("self_attn.v_proj")
        self.wo = lin("self_attn.o_proj")
        self.wg = lin("mlp.gate_proj")
        self.wu = lin("mlp.up_proj")
        self.wd = lin("mlp.down_proj")
        self.rot = nn.Parameter(rotate_half_matrix(cfg.hd), requires_grad=False)
        # conv=True: linear layers as 1x1 convolutions. cv18xx quantizes conv
        # weights per output channel, MatMul weights per tensor.
        self.conv = False

    def lin(self, x, w):
        if not self.conv:
            return x @ w
        y = nn.functional.conv2d(x.reshape(1, -1, 1, 1), w.t().reshape(w.shape[1], w.shape[0], 1, 1))
        return y.reshape(1, 1, -1)

    def forward(self, h, cos, sin, k_cache, v_cache, mask):
        c = self.cfg
        g = c.nh // c.nkv
        x = self.norm1(h)
        q = self.lin(x, self.wq).reshape(1, c.nkv, g, c.hd)  # head = kv * g + group
        k = self.lin(x, self.wk).reshape(1, c.nkv, 1, c.hd)
        v = self.lin(x, self.wv).reshape(1, c.nkv, 1, c.hd)
        q = q * cos + (q @ self.rot) * sin
        k = k * cos + (k @ self.rot) * sin
        keys = torch.cat([k_cache, k], dim=2)  # [1,nkv,S+1,hd]
        vals = torch.cat([v_cache, v], dim=2)
        att = torch.softmax(q @ keys.transpose(2, 3) + mask, dim=-1)  # [1,nkv,g,S+1]
        o = self.lin((att @ vals).reshape(1, 1, c.d), self.wo)
        h2 = h + o
        x = self.norm2(h2)
        m = self.lin(nn.functional.silu(self.lin(x, self.wg)) * self.lin(x, self.wu), self.wd)
        return (o + m if self.delta else h2 + m), k, v


class DecodeLayerT(nn.Module):
    """T tokens per call (chunked prefill, draft verification), weights of a
    DecodeLayer. Only 4-D reshapes/permutes; RoPE tables and mask come in at
    full shape (no broadcast). Rows of q are ordered (token, group).
    Inputs: h [1,T,d], cos_q/sin_q [1,nkv,T*g,hd], cos_k/sin_k [1,nkv,T,hd],
    k_cache/v_cache [1,nkv,S,hd], mask [1,nkv,T*g,S+T].
    Outputs: dh [1,T,d], k_new/v_new [1,nkv,T,hd]."""

    def __init__(self, layer, T):
        super().__init__()
        self.l = layer
        self.T = T

    def forward(self, h, cos_q, sin_q, cos_k, sin_k, k_cache, v_cache, mask):
        c, l, T = self.l.cfg, self.l, self.T
        g = c.nh // c.nkv
        x = l.norm1(h)
        q = l.lin(x, l.wq).reshape(1, T, c.nkv, g * c.hd).permute(0, 2, 1, 3).reshape(1, c.nkv, T * g, c.hd)
        k = l.lin(x, l.wk).reshape(1, T, c.nkv, c.hd).permute(0, 2, 1, 3)
        v = l.lin(x, l.wv).reshape(1, T, c.nkv, c.hd).permute(0, 2, 1, 3)
        q = q * cos_q + (q @ l.rot) * sin_q
        k = k * cos_k + (k @ l.rot) * sin_k
        # Cache and new keys apart: a concat would copy the whole cache every call.
        S = k_cache.shape[2]
        s = torch.cat([q @ k_cache.transpose(2, 3), q @ k.transpose(2, 3)], dim=3)  # [1,nkv,T*g,S+T]
        att = torch.softmax(s + mask, dim=-1)
        o = att[..., :S] @ v_cache + att[..., S:] @ v
        o = o.reshape(1, c.nkv, T, g * c.hd).permute(0, 2, 1, 3).reshape(1, T, c.d)
        o = l.lin(o, l.wo)
        x = l.norm2(h + o)
        m = l.lin(nn.functional.silu(l.lin(x, l.wg)) * l.lin(x, l.wu), l.wd)
        return o + m, k, v


def multi_inputs(cfg, pos, T, S):
    """RoPE tables and mask for tokens at positions pos..pos+T-1; cache
    slots 0..pos-1 hold earlier tokens."""
    g = cfg.nh // cfg.nkv
    cos, sin = rope_tables(cfg, list(range(pos, pos + T)))  # [T, hd]
    cos_q = cos.repeat_interleave(g, 0).expand(cfg.nkv, T * g, cfg.hd).reshape(1, cfg.nkv, T * g, cfg.hd)
    sin_q = sin.repeat_interleave(g, 0).expand(cfg.nkv, T * g, cfg.hd).reshape(1, cfg.nkv, T * g, cfg.hd)
    cos_k = cos.expand(cfg.nkv, T, cfg.hd).reshape(1, cfg.nkv, T, cfg.hd)
    sin_k = sin.expand(cfg.nkv, T, cfg.hd).reshape(1, cfg.nkv, T, cfg.hd)
    mask = torch.full((1, cfg.nkv, T * g, S + T), NEG)
    mask[..., :pos] = 0
    for t in range(T):
        mask[:, :, t * g:(t + 1) * g, S:S + t + 1] = 0
    return cos_q.contiguous(), sin_q.contiguous(), cos_k.contiguous(), sin_k.contiguous(), mask


class Decoder:
    """Full model on the CPU from DecodeLayer modules (reference + calibration)."""

    def __init__(self, path, S):
        self.cfg = cfg = Config(path)
        w = load_weights(path)
        self.emb = w["model.embed_tokens.weight"]
        self.norm = RMSNorm(w["model.norm.weight"], cfg.eps)
        self.layers = [DecodeLayer(cfg, w, i) for i in range(cfg.layers)]
        self.S = S
        self.reset()

    def reset(self):
        c = self.cfg
        self.k = torch.zeros(c.layers, 1, c.nkv, self.S, c.hd)
        self.v = torch.zeros_like(self.k)
        self.t = 0

    def inputs(self, token):
        """Per-step inputs shared by all layers."""
        t, S = self.t, self.S
        cos, sin = rope_tables(self.cfg, [t])
        # Full shape, no broadcast: the cv18xx BF16 broadcast add misplaces it.
        c = self.cfg
        mask = torch.full((1, c.nkv, c.nh // c.nkv, S + 1), NEG)
        mask[..., :t] = 0
        mask[..., S] = 0
        return self.emb[token].reshape(1, 1, -1), cos.reshape(1, 1, 1, -1), sin.reshape(1, 1, 1, -1), mask

    @torch.no_grad()
    def step(self, token, record=None):
        h, cos, sin, mask = self.inputs(token)
        for i, layer in enumerate(self.layers):
            if record is not None and i in record:
                record[i].append(dict(h=h.clone(), cos=cos, sin=sin, k_cache=self.k[i].clone(),
                                      v_cache=self.v[i].clone(), mask=mask))
            h, k, v = layer(h, cos, sin, self.k[i], self.v[i], mask)
            self.k[i][:, :, self.t] = k[:, :, 0]
            self.v[i][:, :, self.t] = v[:, :, 0]
        self.t += 1
        return (self.norm(h) @ self.emb.t()).reshape(-1)
