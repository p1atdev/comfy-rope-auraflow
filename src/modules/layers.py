import torch
import torch.nn as nn

import comfy.ops
from comfy.ldm.aura.mmdit import (
    SingleAttention,
    DoubleAttention,
    DiTBlock,
    MMDiTBlock,
    MMDiT,
    modulate,
)
from comfy.ldm.modules.attention import optimized_attention

from .rope import (
    get_image_position_indices,
    get_text_position_indices,
    get_rope_frequencies,
    applye_rope_frequencies,
)


def single_attention_with_rope_forward(self, c: torch.Tensor, rope_freqs: torch.Tensor):
    bsz, seqlen1, _ = c.shape

    q, k, v = self.w1q(c), self.w1k(c), self.w1v(c)
    q = q.view(bsz, seqlen1, self.n_heads, self.head_dim)
    k = k.view(bsz, seqlen1, self.n_heads, self.head_dim)
    v = v.view(bsz, seqlen1, self.n_heads, self.head_dim)
    q, k = self.q_norm1(q), self.k_norm1(k)

    #! apply RoPE
    q, k = (
        applye_rope_frequencies(q, rope_freqs),
        applye_rope_frequencies(k, rope_freqs),
    )

    output = optimized_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        self.n_heads,
        skip_reshape=True,
    )
    c = self.w1o(output)
    return c


def double_attention_with_rope_forward(
    self, c: torch.Tensor, x: torch.Tensor, rope_freqs: torch.Tensor
):
    bsz, seqlen1, _ = c.shape
    bsz, seqlen2, _ = x.shape

    cq, ck, cv = self.w1q(c), self.w1k(c), self.w1v(c)
    cq = cq.view(bsz, seqlen1, self.n_heads, self.head_dim)
    ck = ck.view(bsz, seqlen1, self.n_heads, self.head_dim)
    cv = cv.view(bsz, seqlen1, self.n_heads, self.head_dim)
    cq, ck = self.q_norm1(cq), self.k_norm1(ck)

    xq, xk, xv = self.w2q(x), self.w2k(x), self.w2v(x)
    xq = xq.view(bsz, seqlen2, self.n_heads, self.head_dim)
    xk = xk.view(bsz, seqlen2, self.n_heads, self.head_dim)
    xv = xv.view(bsz, seqlen2, self.n_heads, self.head_dim)
    xq, xk = self.q_norm2(xq), self.k_norm2(xk)

    # concat all
    q, k, v = (
        torch.cat([cq, xq], dim=1),
        torch.cat([ck, xk], dim=1),
        torch.cat([cv, xv], dim=1),
    )

    #! apply RoPE
    q, k = (
        applye_rope_frequencies(q, rope_freqs),
        applye_rope_frequencies(k, rope_freqs),
    )

    output = optimized_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        self.n_heads,
        skip_reshape=True,
    )

    c, x = output.split([seqlen1, seqlen2], dim=1)
    c = self.w1o(c)
    x = self.w2o(x)

    return c, x


def dit_block_with_rope_forward(
    self,
    cx: torch.Tensor,
    global_cond: torch.Tensor,
    rope_freqs: torch.Tensor,
    **kwargs,
):
    cxres = cx
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modCX(
        global_cond
    ).chunk(6, dim=1)
    cx = modulate(self.norm1(cx), shift_msa, scale_msa)
    cx = self.attn(cx, rope_freqs)  #! pass rope_freqs to attention
    cx = self.norm2(cxres + gate_msa.unsqueeze(1) * cx)
    mlpout = self.mlp(modulate(cx, shift_mlp, scale_mlp))
    cx = gate_mlp.unsqueeze(1) * mlpout

    cx = cxres + cx

    return cx


def mmdit_block_with_rope_forward(
    self,
    c: torch.Tensor,
    x: torch.Tensor,
    global_cond: torch.Tensor,
    rope_freqs: torch.Tensor,
    **kwargs,
):
    cres, xres = c, x

    cshift_msa, cscale_msa, cgate_msa, cshift_mlp, cscale_mlp, cgate_mlp = self.modC(
        global_cond
    ).chunk(6, dim=1)

    c = modulate(self.normC1(c), cshift_msa, cscale_msa)

    # xpath
    xshift_msa, xscale_msa, xgate_msa, xshift_mlp, xscale_mlp, xgate_mlp = self.modX(
        global_cond
    ).chunk(6, dim=1)

    x = modulate(self.normX1(x), xshift_msa, xscale_msa)

    # attention
    c, x = self.attn(c, x, rope_freqs)  #! pass rope_freqs to attention

    c = self.normC2(cres + cgate_msa.unsqueeze(1) * c)
    c = cgate_mlp.unsqueeze(1) * self.mlpC(modulate(c, cshift_mlp, cscale_mlp))
    c = cres + c

    x = self.normX2(xres + xgate_msa.unsqueeze(1) * x)
    x = xgate_mlp.unsqueeze(1) * self.mlpX(modulate(x, xshift_mlp, xscale_mlp))
    x = xres + x

    return c, x


def mmdit_with_rope_forward(self, x, t, context, transformer_options={}, **kwargs):
    patches_replace = transformer_options.get("patches_replace", {})
    #! RoPE args
    dim_sizes = transformer_options.get("dim_sizes", [32, 112, 112])
    rope_theta = transformer_options.get("rope_theta", 10000)

    # patchify x, add PE
    b, c, h, w = x.shape

    # pe_indexes = self.pe_selection_index_based_on_dim(h, w)
    # print(pe_indexes, pe_indexes.shape)

    x = self.init_x_linear(self.patchify(x))  # B, T_x, D
    #! not to apply learned positional embeddings
    # x = self.apply_pos_embeds(x, h, w)
    # x = x + self.positional_encoding[:, : x.size(1)].to(device=x.device, dtype=x.dtype)
    # x = x + self.positional_encoding[:, pe_indexes].to(device=x.device, dtype=x.dtype)

    # process conditions for MMDiT Blocks
    c_seq = context  # B, T_c, D_c
    # t = timestep

    c = self.cond_seq_linear(c_seq)  # B, T_c, D
    c = torch.cat(
        [
            comfy.ops.cast_to_input(self.register_tokens, c).repeat(c.size(0), 1, 1),
            c,
        ],
        dim=1,
    )

    #! instead, we prepare RoPE
    text_indices = get_text_position_indices(c.size(1), dim_sizes)
    image_indices = get_image_position_indices(h, w)
    rope_freqs = get_rope_frequencies(
        torch.cat([text_indices, image_indices], dim=0), dim_sizes, rope_theta
    )

    global_cond = self.t_embedder(t, x.dtype)  # B, D

    blocks_replace = patches_replace.get("dit", {})
    if len(self.double_layers) > 0:
        for i, layer in enumerate(self.double_layers):
            if ("double_block", i) in blocks_replace:

                def block_wrap(args):
                    out = {}
                    out["txt"], out["img"] = layer(
                        args["txt"], args["img"], args["vec"]
                    )
                    return out

                out = blocks_replace[("double_block", i)](
                    {"img": x, "txt": c, "vec": global_cond},
                    {"original_block": block_wrap},
                )
                c = out["txt"]
                x = out["img"]
            else:
                c, x = layer(c, x, global_cond, rope_freqs, **kwargs)

    if len(self.single_layers) > 0:
        c_len = c.size(1)
        cx = torch.cat([c, x], dim=1)
        for i, layer in enumerate(self.single_layers):
            if ("single_block", i) in blocks_replace:

                def block_wrap(args):
                    out = {}
                    out["img"] = layer(args["img"], args["vec"])
                    return out

                out = blocks_replace[("single_block", i)](
                    {"img": cx, "vec": global_cond}, {"original_block": block_wrap}
                )
                cx = out["img"]
            else:
                cx = layer(cx, global_cond, rope_freqs, **kwargs)

        x = cx[:, c_len:]

    fshift, fscale = self.modF(global_cond).chunk(2, dim=1)

    x = modulate(x, fshift, fscale)
    x = self.final_linear(x)
    x = self.unpatchify(x, (h + 1) // self.patch_size, (w + 1) // self.patch_size)[
        :, :, :h, :w
    ]
    return x


def replace_to_rope_modules(model: nn.Module):
    def forward_executer(func, module):
        def forward_hook(*args, **kwargs):
            # normal function can't access self, so manually pass module
            return func(module, *args, **kwargs)

        return forward_hook

    for _name, module in model.named_modules():
        if isinstance(module, SingleAttention):
            setattr(
                module,
                "forward",
                forward_executer(single_attention_with_rope_forward, module),
            )
        elif isinstance(module, DoubleAttention):
            setattr(
                module,
                "forward",
                forward_executer(double_attention_with_rope_forward, module),
            )
        elif isinstance(module, DiTBlock):
            setattr(
                module, "forward", forward_executer(dit_block_with_rope_forward, module)
            )
        elif isinstance(module, MMDiTBlock):
            setattr(
                module,
                "forward",
                forward_executer(mmdit_block_with_rope_forward, module),
            )
        elif isinstance(module, MMDiT):
            setattr(
                module, "forward", forward_executer(mmdit_with_rope_forward, module)
            )
