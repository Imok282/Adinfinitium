"""Export a sampled, *real-weight* picture of the trained lander pilot for the website's network drawing.

Run from the repo root (read-only on the checkpoint, writes only website/assets/net.json):
    python website/tools/export_net.py

Source: data/models/lander_pilot.pt -> {"state_dict", "arch", "info"}; arch = PilotFormer(d=384, layers=8, heads=8, crop=32, stem=48)
(worldmodel/lander/pilot.py). Forward pass, verified against the code:
    camera [3 x 32 x 32] -> conv3x3 s1 (48) -> GN,SiLU -> conv3x3 s2 (96) -> GN,SiLU -> conv2x2 s2 (384) = 8x8 = 64 patch tokens (+ sin/cos pos)
    state [24 = 21 sensors + 3 classic ZEM/ZEV commands] -> Linear(24,384) -> SiLU -> Linear(384,384) = 1 state token
    [CLS (learned), state token, 64 patch tokens] = 66 tokens x 384
    -> 8 pre-norm transformer layers (8 heads, MLP 1536, GELU) -> LayerNorm -> CLS token
    -> mu: Linear(384,384), SiLU, Linear(384,3) (tanh -> thrust residual) ;  v: Linear(384,384), SiLU, Linear(384,1) (value)

Columns ("layers" in the JSON) and what one dot means:
    IN camera  : 27 real input pixels = the 3x3 receptive field (3 channels) of the first conv at the image centre, pixel (16,16).
                 neuron index = c*1024 + y*32 + x (flat index into the 3x32x32 image).
    IN state   : all 24 state inputs (indices 0..20 sensors, 21..23 classic commands).
    STEM 48/96 : conv output channels.
    TOKENS     : the 384 embedding channels shared by all 66 tokens.
    Lk attn    : the 384 output channels the attention block of layer k writes into the residual stream.
    Lk MLP     : the 1536 hidden (GELU) units of layer k's MLP.
    HEAD       : 768 = the 384 hidden units of the thrust head (indices 0..383) and of the value head (384..767).
    OUT        : 4 = dthrust x, y, z (0..2) and value (3).
Dots per column: the n units with the largest weight norm (L2 norm of the incoming row + of the outgoing column, each normalised by its
block's mean), all units when the column is small. HEAD is split evenly: top n/2 thrust-head units + top n/2 value-head units.

Edges (the matrix used for each block is stored in "matrix"):
    camera -> STEM48 : stem.0.weight [48,3,3,3], exact (centre position).
    STEM48 -> STEM96 : stem.3.weight summed over its 3x3 kernel (= response to a spatially flat input; GroupNorm/SiLU ignored).
    STEM96 -> TOKENS : stem.6.weight summed over its 2x2 kernel (same caveat).
    state  -> TOKENS : state.2.weight @ state.0.weight (linearised: the SiLU between them is ignored).
    TOKENS -> L1 attn: OV circuit  out_proj.weight @ W_v @ diag(norm1.weight)  (W_v = rows 768..1151 of in_proj_weight).
                       This is the linear path from a token's channels to the attention output; the attention pattern (softmax over Q.K,
                       weights sum to 1) and LayerNorm centring/scaling are not shown. Q/K weights only steer *where* to look.
    Lk attn -> Lk MLP: linear1.weight @ diag(norm2.weight).
    Lk MLP -> Lk+1 attn: OV_{k+1} @ linear2_k  (the MLP writes into the residual stream, the next attention reads it).
    L8 MLP -> HEAD   : [mu.0.weight ; v.0.weight] @ diag(encoder.norm.weight) @ linear2_8.
    HEAD   -> OUT    : block-diagonal [mu.2.weight, v.2.weight], exact.
    Not drawn: residual skip connections (every block also passes its input straight through), biases, CLS parameter, pos-encoding.
Per block we keep the strongest connections by |w| (each sampled dot gets at least its strongest in/out edge, the rest by global |w|),
keep the sign, and store w / max|w| of the kept block (in [-1,1]) plus the raw max_abs. Floats rounded to 3 decimals.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
CKPT = REPO / "data" / "models" / "lander_pilot.pt"
OUT = REPO / "website" / "assets" / "net.json"
N_DOTS = 48          # dots per wide column
K_EDGES = 600        # max edges per block
CAP_BYTES = 1_500_000


def main() -> None:
    st = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {k: v.float() for k, v in st["state_dict"].items()}
    arch, info = st["arch"], st["info"]
    assert arch["kind"] == "former", arch
    d, L, crop, stem = arch["d"], arch["layers"], arch["crop"], arch["stem"]
    params = int(sum(v.numel() for v in st["state_dict"].values()))
    state_dim = sd["state.0.weight"].shape[1]
    ff = sd["encoder.enc.layers.0.linear1.weight"].shape[0]
    side = crop // 4
    n_tok = side * side + 2

    def enc(k: int, name: str) -> str:
        return f"encoder.enc.layers.{k}.{name}"

    def ov(k: int) -> torch.Tensor:                          # [d, d]: residual (pre-norm) -> attention output
        wv = sd[enc(k, "self_attn.in_proj_weight")][2 * d:3 * d]
        return sd[enc(k, "self_attn.out_proj.weight")] @ (wv * sd[enc(k, "norm1.weight")][None, :])

    # camera centre receptive field: conv1 at (16,16), padding 1 -> pixels 15..17
    c0 = crop // 2
    cam_idx = [c * crop * crop + (c0 - 1 + ky) * crop + (c0 - 1 + kx) for c in range(3) for ky in range(3) for kx in range(3)]

    # column specs: id, label, kind, size
    cols = [("IN_camera", f"IN · camera {crop}×{crop}×3", "input", 3 * crop * crop),
            ("IN_state", f"IN · state {state_dim}", "state", state_dim),
            ("STEM_1", f"STEM · conv · {stem}", "stem", stem),
            ("STEM_2", f"STEM · conv · {2 * stem}", "stem", 2 * stem),
            ("TOKENS", f"TOKENS · {n_tok} × {d}", "tokens", d)]
    for k in range(L):
        cols += [(f"L{k + 1}_attn", f"L{k + 1} · attention · {d}", "attn", d), (f"L{k + 1}_mlp", f"L{k + 1} · MLP · {ff}", "mlp", ff)]
    cols += [("HEAD", f"HEAD · {d} thrust + {d} value", "head", 2 * d), ("OUT", "OUT · Δthrust 3 + value", "head", 4)]
    cid = {c[0]: i for i, c in enumerate(cols)}

    # blocks: (from, to, matrix [size_to, size_from] in each column's own index space (camera: 27 local taps), key text)
    blocks = [("IN_camera", "STEM_1", sd["stem.0.weight"].reshape(stem, 27), "stem.0.weight (centre 3×3×3 receptive field, exact)"),
              ("STEM_1", "STEM_2", sd["stem.3.weight"].sum((2, 3)), "stem.3.weight summed over 3×3 kernel"),
              ("STEM_2", "TOKENS", sd["stem.6.weight"].sum((2, 3)), "stem.6.weight summed over 2×2 kernel"),
              ("IN_state", "TOKENS", sd["state.2.weight"] @ sd["state.0.weight"], "state.2.weight @ state.0.weight (SiLU ignored)"),
              ("TOKENS", "L1_attn", ov(0), "encoder.enc.layers.0: out_proj.weight @ in_proj_weight[V] @ diag(norm1.weight) (OV circuit)")]
    for k in range(L):
        blocks.append((f"L{k + 1}_attn", f"L{k + 1}_mlp", sd[enc(k, "linear1.weight")] * sd[enc(k, "norm2.weight")][None, :],
                       f"{enc(k, 'linear1.weight')} @ diag(norm2.weight)"))
        w2 = sd[enc(k, "linear2.weight")]
        if k + 1 < L:
            blocks.append((f"L{k + 1}_mlp", f"L{k + 2}_attn", ov(k + 1) @ w2,
                           f"OV(encoder.enc.layers.{k + 1}) @ {enc(k, 'linear2.weight')}"))
        else:
            h0 = torch.cat([sd["mu.0.weight"], sd["v.0.weight"]], 0) * sd["encoder.norm.weight"][None, :]
            blocks.append((f"L{k + 1}_mlp", "HEAD", h0 @ w2, f"[mu.0.weight; v.0.weight] @ diag(encoder.norm.weight) @ {enc(k, 'linear2.weight')}"))
    out = torch.zeros(4, 2 * d)
    out[:3, :d], out[3:, d:] = sd["mu.2.weight"], sd["v.2.weight"]
    blocks.append(("HEAD", "OUT", out, "block-diag(mu.2.weight, v.2.weight), exact"))

    # dot selection by weight norm (in-row + out-column, each normalised by its block mean)
    local_size = {c[0]: (27 if c[0] == "IN_camera" else c[3]) for c in cols}
    score = {c[0]: torch.zeros(local_size[c[0]]) for c in cols}
    for f, t, m, _ in blocks:
        r, cn = m.norm(dim=1), m.norm(dim=0)
        score[t] += r / r.mean()
        score[f] += cn / cn.mean()
    picked = {}
    for name, s in score.items():
        n = min(N_DOTS, len(s))
        if name == "HEAD":                                  # half thrust-head, half value-head units (value weights are larger)
            sel = torch.cat([torch.topk(s[:d], n // 2).indices, d + torch.topk(s[d:], n - n // 2).indices])
        else:
            sel = torch.topk(s, n).indices
        picked[name] = torch.sort(sel).values

    layers = []
    for name, label, kind, size in cols:
        idx = picked[name].tolist()
        neurons = [cam_idx[i] for i in idx] if name == "IN_camera" else idx
        layers.append({"id": name, "label": label, "kind": kind, "size": size, "n": len(neurons), "neurons": neurons})

    edges = []
    for f, t, m, key in blocks:
        sub = m[picked[t]][:, picked[f]]                    # [n_to, n_from]
        a = sub.abs()
        keep = torch.zeros_like(a, dtype=torch.bool)
        keep[torch.arange(a.shape[0]), a.argmax(1)] = True  # every dst dot: its strongest input
        keep[a.argmax(0), torch.arange(a.shape[1])] = True  # every src dot: its strongest output
        k = min(K_EDGES, a.numel())
        order = torch.argsort(a.flatten(), descending=True)
        flat = keep.flatten()
        for j in order.tolist():
            if int(flat.sum()) >= k:
                break
            flat[j] = True
        keep = flat.view_as(a) & (a > 0)                    # no edge where the true weight is exactly 0 (head block-diagonal)
        di, si = keep.nonzero(as_tuple=True)
        vals = sub[di, si]
        mx = float(vals.abs().max())
        o = torch.argsort(vals.abs())                       # weakest first, so strong lines draw on top
        edges.append({"from": cid[f], "to": cid[t], "src": si[o].tolist(), "dst": di[o].tolist(),
                      "w": [round(float(x) / mx, 3) for x in vals[o]], "max_abs": round(mx, 5), "matrix": key})

    note = ("Real weights of the trained pilot, sampled: per column the units with the largest weight norm, per block the strongest "
            "connections by |w| (sign kept, w normalised by the block's max |w|, raw max in max_abs). Camera dots are the 27 pixels of "
            "the first conv's 3×3×3 receptive field at the image centre. Stem convs 2-3 use kernels summed over space. state->tokens "
            "and the attention 'OV circuit' (out_proj·W_v·LayerNorm gain) are linear compositions; MLP->next attention is OV·linear2; "
            "the head uses [mu.0;v.0]·final-norm gain·linear2. Not drawn: attention pattern (Q·K softmax), residual skips, biases, "
            "nonlinearities (SiLU/GELU) inside compositions, CLS and positional encoding. Each edge's source matrix is in edges[].matrix.")
    doc = {"meta": {"provenance": "modelled", "source": "data/models/lander_pilot.pt", "checkpoint_id": info.get("checkpoint_id", ""),
                    "parameters": params, "frame": "n/a", "units": "normalised weight", "note": note},
           "layers": layers, "edges": edges}
    txt = json.dumps(doc, separators=(",", ":"), ensure_ascii=False)
    assert len(txt.encode()) <= CAP_BYTES, len(txt)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(txt, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}  {len(txt.encode()) / 1024:.0f} KB  params={params:,}  columns={len(layers)}  blocks={len(edges)}")


if __name__ == "__main__":
    main()
