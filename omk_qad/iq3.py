# SPDX-License-Identifier: Apache-2.0
"""IQ3_XXS / IQ3_S as native QAD formats: a CUDA encoder that reproduces llama.cpp's, and their decoder.

The encoder is llama.cpp's ``quantize_row_iq3_xxs_impl`` / ``quantize_row_iq3_s_impl`` (ggml-quants.c) ported
operation by operation, with the lattice tables built exactly as ``iq3xs_init_impl`` builds them:

  * 32-value sub-blocks, values taken by magnitude with per-8 sign bits (IQ3_XXS: an even number of flips, the
    least important value flipped back if needed); weights ``qw * sqrt(sigma2 + x^2)`` (imatrix, E[x^2]) or ``x^2``;
  * 4-value groups snapped to a 256 (IQ3_XXS) / 512 (IQ3_S) point grid of odd levels 1..15: a group off the grid
    takes the best of its precomputed neighbours (2 / 3 nearest distance shells) under ``sqrt(weight)``;
  * 31 (IQ3_XXS) / 19 (IQ3_S) trial scales ``(15 + 0.2 t) / max``, the best weighted fit wins, then a re-snap of
    the off-grid groups (IQ3_S: every group) at the winning scale and a final least-squares scale;
  * 256-value super-block: fp16 ``d = max_scale / 31`` (times 1.0125 / 1.033) and 4-bit sub-block scales.

IQ3_S keeps llama.cpp's write order exactly: an all-zero sub-block does not advance the qs / signs cursor, so the
next sub-blocks are stored one slot early (llama.cpp's own decoder reads them unshifted -- an upstream bug, kept so
the payload is the one llama-quantize writes and the decode is the one llama.cpp serves).

Every sum runs in llama.cpp's order in float32 without fused multiply-adds (``--fmad=false`` plus explicit
round-to-nearest intrinsics), so the bytes match libggml's bit for bit. On a CUDA device encoding ALWAYS uses the
CUDA kernel (no CPU fallback; an extension that cannot be built raises); :func:`encode_reference` is the same
algorithm in torch, for CPU devices and as the kernel's oracle; libggml (``omk_qad.ggml_lib``) is the oracle of both.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import torch

KGRID_256 = (  # kgrid_256: 4 x 3-bit levels per point
    0, 2, 4, 9, 11, 15, 16, 18, 25, 34, 59, 61, 65, 67, 72, 74,
    81, 85, 88, 90, 97, 108, 120, 128, 130, 132, 137, 144, 146, 153, 155, 159,
    169, 175, 189, 193, 199, 200, 202, 213, 248, 267, 287, 292, 303, 315, 317, 321,
    327, 346, 362, 413, 436, 456, 460, 462, 483, 497, 513, 515, 520, 522, 529, 531,
    536, 538, 540, 551, 552, 576, 578, 585, 592, 594, 641, 643, 648, 650, 657, 664,
    698, 704, 706, 720, 729, 742, 758, 769, 773, 808, 848, 852, 870, 889, 901, 978,
    992, 1024, 1026, 1033, 1035, 1040, 1042, 1046, 1049, 1058, 1089, 1091, 1093, 1096, 1098, 1105,
    1112, 1139, 1143, 1144, 1152, 1154, 1161, 1167, 1168, 1170, 1183, 1184, 1197, 1217, 1224, 1228,
    1272, 1276, 1309, 1323, 1347, 1367, 1377, 1404, 1473, 1475, 1486, 1509, 1537, 1544, 1546, 1553,
    1555, 1576, 1589, 1594, 1600, 1602, 1616, 1625, 1636, 1638, 1665, 1667, 1672, 1685, 1706, 1722,
    1737, 1755, 1816, 1831, 1850, 1856, 1862, 1874, 1901, 1932, 1950, 1971, 2011, 2032, 2052, 2063,
    2077, 2079, 2091, 2095, 2172, 2192, 2207, 2208, 2224, 2230, 2247, 2277, 2308, 2345, 2356, 2389,
    2403, 2424, 2501, 2504, 2506, 2520, 2570, 2593, 2616, 2624, 2630, 2646, 2669, 2700, 2714, 2746,
    2754, 2795, 2824, 2835, 2839, 2874, 2882, 2905, 2984, 3028, 3042, 3092, 3108, 3110, 3124, 3153,
    3185, 3215, 3252, 3288, 3294, 3364, 3397, 3434, 3483, 3523, 3537, 3587, 3589, 3591, 3592, 3610,
    3626, 3670, 3680, 3722, 3749, 3754, 3776, 3789, 3803, 3824, 3857, 3873, 3904, 3906, 3924, 3992,
)
KGRID_512 = (  # kgrid_512: 4 x 3-bit levels per point
    0, 1, 2, 5, 7, 8, 9, 10, 12, 14, 16, 17, 21, 27, 32, 34,
    37, 39, 41, 43, 48, 50, 57, 60, 63, 64, 65, 66, 68, 72, 73, 77,
    80, 83, 87, 89, 93, 100, 113, 117, 122, 128, 129, 133, 135, 136, 139, 142,
    145, 149, 152, 156, 162, 165, 167, 169, 171, 184, 187, 195, 201, 205, 208, 210,
    217, 219, 222, 228, 232, 234, 247, 249, 253, 256, 267, 271, 273, 276, 282, 288,
    291, 297, 312, 322, 324, 336, 338, 342, 347, 353, 357, 359, 374, 379, 390, 393,
    395, 409, 426, 441, 448, 450, 452, 464, 466, 470, 475, 488, 492, 512, 513, 514,
    516, 520, 521, 523, 525, 527, 528, 530, 537, 540, 542, 556, 558, 561, 570, 576,
    577, 579, 582, 584, 588, 593, 600, 603, 609, 616, 618, 632, 638, 640, 650, 653,
    655, 656, 660, 666, 672, 675, 685, 688, 698, 705, 708, 711, 712, 715, 721, 727,
    728, 732, 737, 754, 760, 771, 773, 778, 780, 793, 795, 802, 806, 808, 812, 833,
    840, 843, 849, 856, 858, 873, 912, 916, 919, 932, 934, 961, 963, 968, 970, 977,
    989, 993, 1010, 1016, 1024, 1025, 1027, 1029, 1031, 1032, 1034, 1036, 1038, 1041, 1043, 1047,
    1048, 1050, 1057, 1059, 1061, 1064, 1066, 1079, 1080, 1083, 1085, 1088, 1090, 1096, 1099, 1103,
    1106, 1109, 1113, 1116, 1122, 1129, 1153, 1156, 1159, 1169, 1171, 1176, 1183, 1185, 1195, 1199,
    1209, 1212, 1216, 1218, 1221, 1225, 1234, 1236, 1241, 1243, 1250, 1256, 1270, 1281, 1287, 1296,
    1299, 1306, 1309, 1313, 1338, 1341, 1348, 1353, 1362, 1375, 1376, 1387, 1400, 1408, 1410, 1415,
    1425, 1453, 1457, 1477, 1481, 1494, 1496, 1507, 1512, 1538, 1545, 1547, 1549, 1551, 1554, 1561,
    1563, 1565, 1570, 1572, 1575, 1577, 1587, 1593, 1601, 1603, 1605, 1612, 1617, 1619, 1632, 1648,
    1658, 1662, 1664, 1674, 1680, 1690, 1692, 1704, 1729, 1736, 1740, 1745, 1747, 1751, 1752, 1761,
    1763, 1767, 1773, 1787, 1795, 1801, 1806, 1810, 1817, 1834, 1840, 1844, 1857, 1864, 1866, 1877,
    1882, 1892, 1902, 1915, 1934, 1953, 1985, 1987, 2000, 2002, 2013, 2048, 2052, 2058, 2064, 2068,
    2071, 2074, 2081, 2088, 2104, 2114, 2119, 2121, 2123, 2130, 2136, 2141, 2147, 2153, 2157, 2177,
    2179, 2184, 2189, 2193, 2203, 2208, 2223, 2226, 2232, 2244, 2249, 2251, 2256, 2258, 2265, 2269,
    2304, 2306, 2324, 2335, 2336, 2361, 2373, 2375, 2385, 2418, 2443, 2460, 2480, 2504, 2509, 2520,
    2531, 2537, 2562, 2568, 2572, 2578, 2592, 2596, 2599, 2602, 2614, 2620, 2625, 2627, 2629, 2634,
    2641, 2650, 2682, 2688, 2697, 2707, 2712, 2718, 2731, 2754, 2759, 2760, 2775, 2788, 2793, 2805,
    2811, 2817, 2820, 2832, 2842, 2854, 2890, 2902, 2921, 2923, 2978, 3010, 3012, 3026, 3081, 3083,
    3085, 3097, 3099, 3120, 3136, 3152, 3159, 3188, 3210, 3228, 3234, 3245, 3250, 3256, 3264, 3276,
    3281, 3296, 3349, 3363, 3378, 3392, 3395, 3420, 3440, 3461, 3488, 3529, 3531, 3584, 3588, 3591,
    3600, 3602, 3614, 3616, 3628, 3634, 3650, 3657, 3668, 3683, 3685, 3713, 3716, 3720, 3726, 3729,
    3736, 3753, 3778, 3802, 3805, 3819, 3841, 3845, 3851, 3856, 3880, 3922, 3938, 3970, 3993, 4032,
)

SPECS = {"IQ3_XXS": (256, 98), "IQ3_S": (256, 110)}  # type: (values per block, bytes per block)
TYPES = frozenset(SPECS)
SUB = 32
KMAXQ = 8
EPS_XXS = np.float32(1e-8)  # GROUP_MAX_EPS_IQ3_XXS (IQ3_S skips only an exactly-zero sub-block)
_CFG = {  # grid points, distance shells kept per off-grid point, trial scales, fp16 fudge, re-snap every group
    "IQ3_XXS": (256, 2, range(-15, 16), np.float32(1.0125), False),
    "IQ3_S": (512, 3, range(-9, 10), np.float32(1.033), True),
}
_XXS_LEVEL = {1: 4, 3: 12, 5: 20, 7: 28, 9: 36, 11: 44, 13: 52, 15: 62}  # iq3xxs_grid's bytes (15 -> 62, not 60)


@cache
def tables(qtype: str) -> dict[str, np.ndarray]:
    """iq3xs_init_impl: grid (odd levels), kmap (index or -(neighbour offset + 1)), neighbour lists, decoder grid."""
    n, nwant = _CFG[qtype][:2]
    kgrid = KGRID_256 if n == 256 else KGRID_512
    grid = np.array([[2 * ((g >> 3 * i) & 7) + 1 for i in range(4)] for g in kgrid], dtype=np.int32)  # [n, 4]
    kmap = np.full(4096, -1, dtype=np.int32)
    kmap[[sum(((v - 1) // 2) << 3 * i for i, v in enumerate(p)) for p in grid]] = np.arange(n, dtype=np.int32)
    neighbours: list[int] = []
    lists = {}
    for u in range(4096):
        if kmap[u] >= 0:
            continue
        pos = np.array([2 * ((u >> 3 * k) & 7) + 1 for k in range(4)], dtype=np.int32)
        d2 = ((grid - pos) ** 2).sum(1)
        order = np.lexsort((np.arange(n), d2))  # qsort by (d2, index)
        take, cur, nhave = [], d2[order[0]], 1
        for j in order:
            if d2[j] > cur:
                if nhave == nwant:
                    break
                cur, nhave = d2[j], nhave + 1
            take.append(int(j))
        kmap[u] = -(len(neighbours) + 1)
        neighbours += [len(take), *take]
        lists[u] = take
    pad = np.full((4096, max(len(t) for t in lists.values())), -1, dtype=np.int64)
    for u, t in lists.items():
        pad[u, : len(t)] = t
    dec = grid if qtype == "IQ3_S" else np.vectorize(_XXS_LEVEL.get)(grid)
    signs = np.array([s | ((s.bit_count() & 1) << 7) for s in range(128)], dtype=np.int64)  # ksigns_iq2xs
    return {"grid": grid, "kmap": kmap, "neighbours": np.array(neighbours, dtype=np.uint16), "pad": pad,
            "dec": dec.astype(np.float32), "ksigns": signs}


def _dev_tables(qtype: str, device: torch.device) -> dict[str, torch.Tensor]:
    key = (qtype, str(device))
    cache = _dev_tables.__dict__.setdefault("cache", {})
    if key not in cache:
        t = tables(qtype)
        cache[key] = {
            "grid": torch.from_numpy(t["grid"].astype(np.float32)).to(device),
            "gridl": torch.from_numpy((t["grid"].astype(np.int64) - 1) // 2).to(device),
            "kmap": torch.from_numpy(t["kmap"].astype(np.int64)).to(device),
            "pad": torch.from_numpy(t["pad"]).to(device),
            "dec": torch.from_numpy(t["dec"]).to(device),
            "ksigns": torch.from_numpy(t["ksigns"]).to(device),
            # kernel inputs: packed int8 grid (as llama.cpp's uint32), int32 kmap, uint16 neighbours (as int32)
            "kgrid_u32": torch.from_numpy(
                (t["grid"].astype(np.uint32) << (8 * np.arange(4, dtype=np.uint32))).sum(1).astype(np.int64)
                .astype(np.uint32).view(np.int32)).to(device),
            "kmap_i32": torch.from_numpy(t["kmap"]).to(device),
            "neigh_i32": torch.from_numpy(t["neighbours"].astype(np.int32)).to(device),
        }
    return cache[key]


def _ext():
    """The CUDA encoder (built on first use); raises if it cannot be built -- there is no CPU fallback."""
    if not hasattr(_ext, "mod"):
        from modelopt.torch.utils.cpp_extension import load_cpp_extension

        here = Path(__file__).parent / "kernels"
        _ext.mod = load_cpp_extension(
            name="omk_cuda_ext_iq3",
            sources=[here / "iq3.cpp", here / "iq3.cu"],
            cuda_version_specifiers=">=11.8",
            fail_msg="omk_qad IQ3_XXS/IQ3_S CUDA encoder is unavailable (no CPU fallback on a CUDA device).",
            extra_cuda_cflags=["-O3", "--fmad=false"],
            raise_if_failed=True,
        )
    return _ext.mod


def _check(qtype: str, x: torch.Tensor, importance: torch.Tensor | None) -> None:
    if qtype not in SPECS:
        raise ValueError(f"not an IQ3 type: {qtype}")
    if x.dim() != 2 or x.dtype != torch.float32 or x.shape[1] % 256:
        raise ValueError(f"{qtype} needs float32 [rows, cols] with cols a multiple of 256, got {x.dtype} "
                         f"{tuple(x.shape)}")
    if importance is not None and (importance.dim() != 1 or importance.numel() != x.shape[1]):
        raise ValueError(f"importance must be [cols={x.shape[1]}], got {tuple(importance.shape)}")


def encode(qtype: str, x: torch.Tensor, importance: torch.Tensor | None = None) -> torch.Tensor:
    """uint8 payload [rows, cols / 256, block bytes] of float32 ``x`` [rows, cols]; ``importance`` = E[x^2] [cols]."""
    _check(qtype, x, importance)
    if not x.is_cuda:
        return encode_reference(qtype, x, importance)
    t = _dev_tables(qtype, x.device)
    imp = None if importance is None else importance.to(device=x.device, dtype=torch.float32).contiguous()
    return _ext().iq3_pack(x.contiguous(), imp, qtype == "IQ3_S", t["kgrid_u32"], t["kmap_i32"], t["neigh_i32"])


def decode(qtype: str, packed: torch.Tensor, rows: int, cols: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[rows, cols] values of a payload (llama.cpp's dequantize_row_iq3_xxs / _s), on the payload's device."""
    nbytes = SPECS[qtype][1]
    p = packed.reshape(-1, nbytes)
    t = _dev_tables(qtype, p.device)
    d = p[:, :2].contiguous().view(torch.float16).float()  # [B, 1]
    bit = torch.arange(8, device=p.device)
    if qtype == "IQ3_XXS":
        qs = p[:, 2:66].long().reshape(-1, 8, 8)  # [B, ib, 8 grid indices]
        sas = p[:, 66:98].contiguous().view(torch.int32).long() & 0xFFFFFFFF  # [B, 8]
        db = (d * ((sas >> 28).float() + 0.5)) * 0.5  # [B, 8]
        sg = t["ksigns"][(sas[:, :, None] >> (7 * torch.arange(4, device=p.device))) & 127]  # [B, 8, 4]
    else:
        qs = p[:, 2:66].long().reshape(-1, 8, 8)
        qh = p[:, 66:74].long()  # [B, 8]
        qs = qs | (((qh[:, :, None] >> bit) & 1) << 8)
        sg = p[:, 74:106].long().reshape(-1, 8, 4)
        sc = p[:, 106:110].long()
        ls = torch.stack([sc & 0xF, sc >> 4], 2).reshape(-1, 8)  # sub-block ib -> nibble ib % 2 of byte ib // 2
        db = d * (1 + 2 * ls).float()
    vals = t["dec"][qs].reshape(-1, 8, 4, 8)  # [B, ib, l, 8 values (two grid points)]
    neg = (sg[..., None] >> bit) & 1  # [B, ib, l, 8]
    y = (db[:, :, None, None] * vals) * (1 - 2 * neg).float()
    return y.reshape(rows, cols).to(dtype)


# ------------------------------------------------------------------------------------------- torch reference
def _f32(v, like: torch.Tensor) -> torch.Tensor:
    return torch.full_like(like, float(np.float32(v)))


def _snap(t: dict, u: torch.Tensor, xv: torch.Tensor, wa: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """iq3_find_best_neighbour for groups whose code u is off the grid: grid index of the best neighbour."""
    cand = t["pad"][u]  # [G, maxn]
    q = t["grid"][cand.clamp(min=0)]  # [G, maxn, 4]
    d2 = torch.zeros(cand.shape, dtype=torch.float32, device=u.device)
    for i in range(4):
        diff = scale[:, None] * q[:, :, i] - xv[:, None, i]
        d2 = d2 + (wa[:, None, i] * diff) * diff
    d2 = torch.where(cand >= 0, d2, torch.full_like(d2, float("inf")))
    return cand.gather(1, d2.argmin(1, keepdim=True))[:, 0]  # first minimum (strict <)


def _levels(t: dict, inv: torch.Tensor, xval: torch.Tensor, wa: torch.Tensor, scale: torch.Tensor):
    """Levels [S, 32] at inverse scale ``inv`` [S] with off-grid groups snapped at ``scale``; on-grid mask [S, 8]."""
    lev = torch.round((inv[:, None] * xval - 1) * 0.5).clamp(0, KMAXQ - 1).long()  # nearest_int(0.5f*(id*x - 1))
    l4 = lev.reshape(-1, 8, 4)
    u = (l4 << (3 * torch.arange(4, device=lev.device))).sum(-1)  # [S, 8]
    on = t["kmap"][u] >= 0
    off = (~on).nonzero(as_tuple=True)
    if off[0].numel():
        g = _snap(t, u[off], xval.reshape(-1, 8, 4)[off], wa.reshape(-1, 8, 4)[off], scale[off[0]])
        l4 = l4.clone()
        l4[off] = t["gridl"][g]
    return l4.reshape(-1, SUB), on


def _sums(w: torch.Tensor, xval: torch.Tensor, lev: torch.Tensor):
    q = (2 * lev + 1).float()
    sumqx = torch.zeros(w.shape[0], dtype=torch.float32, device=w.device)
    sumq2 = torch.zeros_like(sumqx)
    for i in range(SUB):
        sumqx = sumqx + (w[:, i] * xval[:, i]) * q[:, i]
        sumq2 = sumq2 + (w[:, i] * q[:, i]) * q[:, i]
    return sumqx, sumq2


def _search(qtype: str, xs: torch.Tensor, w: torch.Tensor):
    """Every 32-value sub-block [S, 32]: grid indices [S, 8], scale [S], sign bytes [S, 4], skipped mask [S]."""
    _, _, trials, _, resnap_all = _CFG[qtype]
    t = _dev_tables(qtype, xs.device)
    ns = xs.shape[0]
    wa = torch.sqrt(w.double()).float()  # correctly rounded sqrtf (torch's float32 CPU sqrt is not always)
    neg = ~(xs >= 0)
    xval = torch.where(neg, -xs, xs)
    bits = 1 << torch.arange(8, device=xs.device)
    sgn = (neg.reshape(ns, 4, 8).long() * bits).sum(-1)  # [S, 4]
    if qtype == "IQ3_XXS":  # an even number of flips: flip back the least important value of an odd group
        odd = neg.reshape(ns, 4, 8).sum(-1) % 2 == 1
        imin = ((w * xs) * xs).reshape(ns, 4, 8).argmin(-1)  # first minimum (strict <)
        flat = (torch.arange(4, device=xs.device) * 8 + imin)[odd]
        rows = torch.arange(ns, device=xs.device)[:, None].expand(ns, 4)[odd]
        xval = xval.clone()
        xval[rows, flat] = -xval[rows, flat]
        sgn = torch.where(odd, sgn ^ (1 << imin), sgn) & 127
    mx = xval.amax(1)
    skip = mx < torch.tensor(EPS_XXS, device=xs.device) if qtype == "IQ3_XXS" else mx == 0
    mxs = torch.where(skip, torch.ones_like(mx), mx)  # keeps divisions finite; skipped rows are masked below
    best = torch.zeros_like(mx)
    scale = mxs / _f32(2 * KMAXQ - 1, mx)
    lev = torch.zeros(ns, SUB, dtype=torch.long, device=xs.device)
    on = torch.full((ns, 8), qtype == "IQ3_XXS", device=xs.device)
    for it in trials:
        inv = _f32(np.float32(2 * KMAXQ - 1) + np.float32(it) * np.float32(0.2), mx) / mxs
        this = torch.ones_like(inv) / inv
        la, ona = _levels(t, inv, xval, wa, this)
        sumqx, sumq2 = _sums(w, xval, la)
        better = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
        nd = sumqx / sumq2
        scale = torch.where(better, nd, scale)
        best = torch.where(better, nd * sumqx, best)
        lev = torch.where(better[:, None], la, lev)
        on = torch.where(better[:, None], ona, on)
    need = (~on).any(1) & (scale > 0) & ~skip
    if need.any():
        sc = torch.where(need, scale, torch.ones_like(scale))
        la, _ = _levels(t, torch.ones_like(sc) / sc, xval, wa, sc)
        redo = need[:, None].expand(-1, 8) if resnap_all else need[:, None] & ~on  # XXS: only the off-grid groups
        lev = torch.where(redo.repeat_interleave(4, 1), la, lev)
        sumqx, sumq2 = _sums(w, xval, lev)
        scale = torch.where(need & (sumq2 > 0), sumqx / sumq2, scale)
    flip = scale < 0  # "should never happen": positive scale, flipped signs
    scale = torch.where(flip, -scale, scale)
    sgn = torch.where(flip[:, None], ~sgn & (127 if qtype == "IQ3_XXS" else 255), sgn)
    scale = torch.where(skip, torch.zeros_like(scale), scale)
    lev = torch.where(skip[:, None], torch.zeros_like(lev), lev)
    u = (lev.reshape(ns, 8, 4) << (3 * torch.arange(4, device=xs.device))).sum(-1)
    gi = t["kmap"][u]
    assert bool((gi >= 0).all()), "a group is off the grid (llama.cpp aborts here)"
    return gi, scale, sgn, skip


def _weights(x: torch.Tensor, imp: torch.Tensor | None) -> torch.Tensor:
    """Per-value weights of super-blocks x [B, 256] (imp: [B, 256] importance columns or None)."""
    if imp is None:
        return x * x
    sigma2 = torch.zeros(x.shape[0], dtype=torch.float32, device=x.device)
    for j in range(256):
        sigma2 = sigma2 + x[:, j] * x[:, j]
    sigma2 = (2 * sigma2) / _f32(256, sigma2)
    return imp * torch.sqrt((sigma2[:, None] + x * x).double()).float()


def encode_reference(qtype: str, x: torch.Tensor, importance: torch.Tensor | None = None) -> torch.Tensor:
    """The CUDA kernel's algorithm in torch (any device), byte-identical to it and to libggml."""
    _check(qtype, x, importance)
    rows, cols = x.shape
    nbytes = SPECS[qtype][1]
    fudge = _CFG[qtype][3]
    xb = x.reshape(-1, 256)
    nb = xb.shape[0]
    imp = None
    if importance is not None:
        imp = importance.to(device=x.device, dtype=torch.float32).reshape(1, cols // 256, 256)
        imp = imp.expand(rows, -1, -1).reshape(-1, 256)
    w = _weights(xb, imp)
    gi, scale, sgn, skip = _search(qtype, xb.reshape(-1, SUB), w.reshape(-1, SUB))
    gi, scale, sgn, skip = gi.reshape(nb, 8, 8), scale.reshape(nb, 8), sgn.reshape(nb, 8, 4), skip.reshape(nb, 8)
    out = torch.zeros(nb, nbytes, dtype=torch.uint8, device=x.device)
    max_scale = scale.amax(1)
    live = max_scale > 0
    ms = torch.where(live, max_scale, torch.ones_like(max_scale))
    d = ms / _f32(31, ms)
    dh = torch.where(live, d * _f32(fudge, d), torch.zeros_like(d))
    out[:, :2] = dh.to(torch.float16).reshape(-1, 1).view(torch.uint8)
    inv = torch.ones_like(d) / d
    lsc = torch.round((inv[:, None] * scale - 1) * 0.5).clamp(0, 15).long()  # [nb, 8]
    if qtype == "IQ3_XXS":
        sas = sgn[:, :, 0] | (sgn[:, :, 1] << 7) | (sgn[:, :, 2] << 14) | (sgn[:, :, 3] << 21) | (lsc << 28)
        sas = torch.where(skip, lsc << 28, sas)  # a skipped sub-block stored no signs
        q3 = torch.where(skip[:, :, None], torch.zeros_like(gi), gi).reshape(nb, 64)
        body = torch.cat([q3, torch.stack([(sas >> s) & 0xFF for s in (0, 8, 16, 24)], 2).reshape(nb, 32)], 1)
        out[:, 2:] = torch.where(live[:, None], body, torch.zeros_like(body)).to(torch.uint8)
        return out.reshape(rows, cols // 256, nbytes)
    # IQ3_S: qs / signs at the running cursor (a skipped sub-block does not advance it), qh at the sub-block index
    keep = ~skip
    rank = torch.cumsum(keep.long(), 1) - keep.long()  # [nb, 8]
    b_idx, ib = keep.nonzero(as_tuple=True)
    k = torch.arange(8, device=x.device)
    qs_pos = 2 + rank[b_idx, ib][:, None] * 8 + k  # [N, 8]
    out[b_idx[:, None].expand(-1, 8), qs_pos] = (gi[b_idx, ib] & 255).to(torch.uint8)
    sg_pos = 74 + rank[b_idx, ib][:, None] * 4 + torch.arange(4, device=x.device)
    out[b_idx[:, None].expand(-1, 4), sg_pos] = sgn[b_idx, ib].to(torch.uint8)
    qh = (((gi >> 8) & 1) << k).sum(-1)  # [nb, 8]
    out[:, 66:74] = torch.where(keep, qh, torch.zeros_like(qh)).to(torch.uint8)
    pairs = (lsc[:, 0::2] | (lsc[:, 1::2] << 4))
    out[:, 106:110] = torch.where(live[:, None], pairs, torch.zeros_like(pairs)).to(torch.uint8)
    return out.reshape(rows, cols // 256, nbytes)
