import torch
import triton
import triton.language as tl

"""GROUP_NM_SWEEP = [4, 8]
NUM_STAGES_SWEEP = [2, 3, 4]
NUM_WARPS_SWEEP = [2, 4]"""

GROUP_NM_SWEEP = [4]
NUM_STAGES_SWEEP = [3]
NUM_WARPS_SWEEP = [2]
KEY_CACHE = ["BATCH_SIZE", "NUM_HEADS", "SEQ_LEN", "HEAD_DIM"]

def _build_axial_rope(
    side_len: int,
    head_dim: int,
    device: torch.device,
    base: float = 10000.0,
):
    """
    2-D axial RoPE (Rotary Positional Embedding) tables for an N x N grid.
    Returns (cos_x, sin_x, cos_y, sin_y), each [N*N, head_dim//2].
    """
    assert head_dim % 4 == 0
    N = side_len
    per_axis = head_dim // 2
    half = per_axis // 2  # distinct angles (pairs) per axis

    # frequency ladder ω_k
    inv_freq = base ** (-torch.arange(0, half, device=device) / half)  # [half]

    # 1-D positions reused for both axes
    pos = torch.arange(N, device=device)  # [N]
    theta = torch.outer(pos, inv_freq)  # [N, half]

    # duplicate each angle for pairwise rotation -> [N, per_axis]
    cos_1d = torch.cos(theta).repeat_interleave(2, dim=-1)
    sin_1d = torch.sin(theta).repeat_interleave(2, dim=-1)

    # broadcast to 2-D grid and flatten with lin = y*N + x
    cos_x = cos_1d[:, None, :].expand(N, N, per_axis).reshape(N * N, per_axis)
    sin_x = sin_1d[:, None, :].expand(N, N, per_axis).reshape(N * N, per_axis)
    cos_y = cos_1d[None, :, :].expand(N, N, per_axis).reshape(N * N, per_axis)
    sin_y = sin_1d[None, :, :].expand(N, N, per_axis).reshape(N * N, per_axis)
    return cos_x, sin_x, cos_y, sin_y


def _sdpa_comp_dtype(x: torch.Tensor) -> torch.dtype:
    dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else x.dtype
    if dtype is torch.float16:
        return tl.float16
    if dtype is torch.bfloat16:
        return tl.bfloat16
    if dtype is torch.float32:
        return tl.float32
    raise dtype


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_Q": BLOCK_Q, "BLOCK_KV": BLOCK_KV, "GROUP_M": GROUP_M},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_Q in [64, 128]
        for BLOCK_KV in [32, 64]
        for GROUP_M in GROUP_NM_SWEEP
        for num_stages in NUM_STAGES_SWEEP
        for num_warps in NUM_WARPS_SWEEP
    ],
    key=KEY_CACHE,
)
@triton.jit
def _attn_fwd(
    Q, K, V, M, O,                 # Q,K,V:[B,H,S,D]; M:[B,H,S] (fp32); O:[B,H,S,D]
    # Strides (inputs)
    sqb, sqh, sqs, sqd,
    skb, skh, sks, skd,
    svb, svh, svs, svd,
    sob, soh, sos, sod,
    # Axial RoPE tables for Q & K (each [N,P] where N=H_img*W_img, P=D/4)
    COSX, SINX, COSY, SINY,
    cosx_s, cosx_p,
    sinx_s, sinx_p,
    cosy_s, cosy_p,
    siny_s, siny_p,
    # Meta
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    D2: tl.constexpr,                # HEAD_DIM // 2
    P: tl.constexpr,                 # HEAD_DIM // 4
    HAS_CLS: tl.constexpr,           # 1 if CLS at index 0
    softmax_scale: tl.constexpr,     # 1/sqrt(D)
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    DTYPE: tl.constexpr,             # compute dtype (e.g., tl.float32 or tl.float16)
    GROUP_M: tl.constexpr,
):
    tl.static_assert((HEAD_DIM % 4) == 0)

    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # --- no swizzle: plain linear tiling over M ---
    start_q = pid_m * BLOCK_Q
    if start_q >= SEQ_LEN:
        return
    # ---- (b,h) plane selection ----
    b = pid_bh // NUM_HEADS
    h = pid_bh %  NUM_HEADS

    off_bh_q = (b * sqb + h * sqh).to(tl.int32)
    off_bh_k = (b * skb + h * skh).to(tl.int32)
    off_bh_v = (b * svb + h * svh).to(tl.int32)
    off_bh_o = (b * sob + h * soh).to(tl.int32)

    # ---- indices/consts ----
    rows    = start_q + tl.arange(0, BLOCK_Q)            # [BQ]
    rows64  = rows.to(tl.int32)
    q_valid = rows < SEQ_LEN
    cols    = tl.arange(0, BLOCK_KV)                     # [BKV]
    pair_ix = tl.arange(0, P).to(tl.int32)

    sqs_i = tl.full((1,), sqs, tl.int32)
    sqd_i = tl.full((1,), sqd, tl.int32)
    sks_i = tl.full((1,), sks, tl.int32)
    skd_i = tl.full((1,), skd, tl.int32)
    svs_i = tl.full((1,), svs, tl.int32)
    svd_i = tl.full((1,), svd, tl.int32)
    D2_i  = tl.full((1,), D2,  tl.int32)

    even = (2 * pair_ix)
    odd  = (2 * pair_ix + 1)

    # ---- gather Q pairs [BQ,P] ----
    base_Q = Q + off_bh_q
    row_off_q = rows64[:, None] * sqs_i
    qcol_xe = even * sqd_i
    qcol_xo = odd  * sqd_i
    qcol_ye = (D2_i + even) * sqd_i
    qcol_yo = (D2_i + odd ) * sqd_i

    Qxe = tl.load(base_Q + row_off_q + qcol_xe[None,:], mask=q_valid[:,None], other=0.).to(DTYPE)
    Qxo = tl.load(base_Q + row_off_q + qcol_xo[None,:], mask=q_valid[:,None], other=0.).to(DTYPE)
    Qye = tl.load(base_Q + row_off_q + qcol_ye[None,:], mask=q_valid[:,None], other=0.).to(DTYPE)
    Qyo = tl.load(base_Q + row_off_q + qcol_yo[None,:], mask=q_valid[:,None], other=0.).to(DTYPE)

    # ---- Q-side RoPE ([N,P] tables) ----
    lin_q = rows - HAS_CLS
    lin_q = tl.maximum(lin_q, 0).to(tl.int32)
    is_cls_q = (HAS_CLS != 0) & (rows == 0)

    cx_row = lin_q[:,None] * tl.full((1,), cosx_s, tl.int32)
    sx_row = lin_q[:,None] * tl.full((1,), sinx_s, tl.int32)
    cy_row = lin_q[:,None] * tl.full((1,), cosy_s, tl.int32)
    sy_row = lin_q[:,None] * tl.full((1,), siny_s, tl.int32)

    cx_col = pair_ix[None,:] * tl.full((1,), cosx_p, tl.int32)
    sx_col = pair_ix[None,:] * tl.full((1,), sinx_p, tl.int32)
    cy_col = pair_ix[None,:] * tl.full((1,), cosy_p, tl.int32)
    sy_col = pair_ix[None,:] * tl.full((1,), siny_p, tl.int32)

    CX_q = tl.load(COSX + cx_row + cx_col, mask=q_valid[:,None], other=0.).to(DTYPE)
    SX_q = tl.load(SINX + sx_row + sx_col, mask=q_valid[:,None], other=0.).to(DTYPE)
    CY_q = tl.load(COSY + cy_row + cy_col, mask=q_valid[:,None], other=0.).to(DTYPE)
    SY_q = tl.load(SINY + sy_row + sy_col, mask=q_valid[:,None], other=0.).to(DTYPE)

    Qx_e_r = tl.where(is_cls_q[:,None], Qxe, Qxe*CX_q - Qxo*SX_q)
    Qx_o_r = tl.where(is_cls_q[:,None], Qxo, Qxe*SX_q + Qxo*CX_q)
    Qy_e_r = tl.where(is_cls_q[:,None], Qye, Qye*CY_q - Qyo*SY_q)
    Qy_o_r = tl.where(is_cls_q[:,None], Qyo, Qye*SY_q + Qyo*CY_q)

    # ---- online softmax accumulators ----
    m_i = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_Q,),  0.0,          dtype=tl.float32)
    O_blk = tl.zeros((BLOCK_Q, HEAD_DIM), dtype=tl.float32)

    # ---- KV loop ----
    for start_kv in range(0, SEQ_LEN, BLOCK_KV):
        kv_cols   = start_kv + cols
        kv_valid  = kv_cols < SEQ_LEN
        kv_cols64_k = kv_cols.to(tl.int32)[None,:]
        kv_cols64_v = kv_cols.to(tl.int32)[:, None]

        # gather K pairs -> [P,BKV]
        base_K = K + off_bh_k
        krow_xe = even * skd_i
        krow_xo = odd  * skd_i
        krow_ye = (D2_i + even) * skd_i
        krow_yo = (D2_i + odd ) * skd_i

        Kxe = tl.load(base_K + krow_xe[:,None] + kv_cols64_k * sks_i, mask=kv_valid[None,:], other=0.).to(DTYPE)
        Kxo = tl.load(base_K + krow_xo[:,None] + kv_cols64_k * sks_i, mask=kv_valid[None,:], other=0.).to(DTYPE)
        Kye = tl.load(base_K + krow_ye[:,None] + kv_cols64_k * sks_i, mask=kv_valid[None,:], other=0.).to(DTYPE)
        Kyo = tl.load(base_K + krow_yo[:,None] + kv_cols64_k * sks_i, mask=kv_valid[None,:], other=0.).to(DTYPE)

        # K-side RoPE (pair-major [P,BKV]); CLS col bypass
        lin_k   = kv_cols - HAS_CLS
        lin_k   = tl.maximum(lin_k, 0).to(tl.int32)
        is_cls_k = (HAS_CLS != 0) & (kv_cols == 0)

        cxk = pair_ix[:,None] * tl.full((1,), cosx_p, tl.int32) + lin_k[None,:] * tl.full((1,), cosx_s, tl.int32)
        sxk = pair_ix[:,None] * tl.full((1,), sinx_p, tl.int32) + lin_k[None,:] * tl.full((1,), sinx_s, tl.int32)
        cyk = pair_ix[:,None] * tl.full((1,), cosy_p, tl.int32) + lin_k[None,:] * tl.full((1,), cosy_s, tl.int32)
        syk = pair_ix[:,None] * tl.full((1,), siny_p, tl.int32) + lin_k[None,:] * tl.full((1,), siny_s, tl.int32)

        CX_k = tl.load(COSX + cxk, mask=kv_valid[None,:], other=0.).to(DTYPE)
        SX_k = tl.load(SINX + sxk, mask=kv_valid[None,:], other=0.).to(DTYPE)
        CY_k = tl.load(COSY + cyk, mask=kv_valid[None,:], other=0.).to(DTYPE)
        SY_k = tl.load(SINY + syk, mask=kv_valid[None,:], other=0.).to(DTYPE)

        Kx_e_r = tl.where(is_cls_k[None,:], Kxe, Kxe*CX_k - Kxo*SX_k)
        Kx_o_r = tl.where(is_cls_k[None,:], Kxo, Kxe*SX_k + Kxo*CX_k)
        Ky_e_r = tl.where(is_cls_k[None,:], Kye, Kye*CY_k - Kyo*SY_k)
        Ky_o_r = tl.where(is_cls_k[None,:], Kyo, Kye*SY_k + Kyo*CY_k)

        # logits tile
        S_tile = (tl.dot(Qx_e_r, Kx_e_r) 
               + tl.dot(Qx_o_r, Kx_o_r) 
               + tl.dot(Qy_e_r, Ky_e_r) 
               + tl.dot(Qy_o_r, Ky_o_r))
        
        S_tile = S_tile * tl.full((1,), softmax_scale, dtype=tl.float32)
        S_tile = tl.where(q_valid[:,None] & kv_valid[None,:], S_tile, -float("inf"))
        
        # online softmax
        m_ij  = tl.maximum(m_i, tl.max(S_tile, axis=1))
        alpha = tl.where(q_valid, tl.exp(m_i - m_ij), 1.0)
        P_blk = tl.where(q_valid[:,None], tl.exp(S_tile - m_ij[:,None]), 0.0)
        l_ij  = tl.sum(P_blk, axis=1)

        # accumulate O
        d_idx  = tl.arange(0, HEAD_DIM)[None, :].to(tl.int32)    # [1, D]
        V_ptrs = (V + off_bh_v) + kv_cols64_v * svs_i + d_idx * svd_i  # [BKV, D]
        V_blk  = tl.load(V_ptrs, mask=kv_valid[:, None], other=0.).to(DTYPE)
        O_blk  = O_blk * alpha[:, None]
        O_blk  = tl.dot(P_blk.to(DTYPE), V_blk, O_blk)                        # (BQ,BKV) @ (BKV,D) + O_blk

        l_i = tl.where(q_valid, l_i * alpha + l_ij, l_i)
        m_i = tl.where(q_valid, m_ij, m_i)
        
        # ---- write back M and O ----
    m_ptrs = M + (b * NUM_HEADS + h) * SEQ_LEN + rows
    tl.store(m_ptrs, m_i + tl.log(l_i + 1e-20), mask=q_valid)

    O_blk = O_blk / l_i[:, None]
    O_ptrs = (O + off_bh_o) + row_off_q + (tl.arange(0, HEAD_DIM)[None, :].to(tl.int32) * tl.full((1,), sod, tl.int32))
    tl.store(O_ptrs, O_blk.to(O.type.element_ty), mask=q_valid[:,None])

@triton.autotune(
    [triton.Config({"BLOCK_Q": bq}, num_stages=ns, num_warps=nw)
     for bq in [32]
     #for bq in [32, 64, 128]
     for ns in NUM_STAGES_SWEEP
     for nw in NUM_WARPS_SWEEP],
    key=KEY_CACHE,
)
@triton.jit
def _attn_bwd_preprocess(
    O, dO, D,
    sOb, sOh, sOs, sOd,          # O strides
    sdb, sdh, sds, sdd,          # dO strides
    NUM_HEADS: tl.constexpr, SEQ_LEN: tl.constexpr,
    BLOCK_Q: tl.constexpr, HEAD_DIM: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
):
    pid_q  = tl.program_id(0)                          # Q-tile id
    pid_bh = tl.program_id(1)                          # packed (batch, head)
    start_q = pid_q * BLOCK_Q
    if start_q >= SEQ_LEN:
        return

    b = pid_bh // NUM_HEADS
    h = pid_bh %  NUM_HEADS
    off_bh_O  = (b * sOb  + h * sOh ).to(tl.int32)
    off_bh_dO = (b * sdb  + h * sdh ).to(tl.int32)

    O_blk = tl.make_block_ptr(
        O + off_bh_O, (SEQ_LEN, HEAD_DIM), (sOs, sOd), (start_q, 0), (BLOCK_Q, HEAD_DIM), (1, 0)
    )
    dO_blk = tl.make_block_ptr(
        dO + off_bh_dO, (SEQ_LEN, HEAD_DIM), (sds, sdd), (start_q, 0), (BLOCK_Q, HEAD_DIM), (1, 0)
    )

    O_block  = tl.load(O_blk,  boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    dO_block = tl.load(dO_blk, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    D_block  = tl.sum(dO_block * O_block, axis=1)

    offs_q = start_q + tl.arange(0, BLOCK_Q)
    tl.store(D + pid_bh * SEQ_LEN + offs_q, D_block, mask=offs_q < SEQ_LEN)

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_Q": BLOCK_Q, "BLOCK_KV": BLOCK_KV, "GROUP_N": GROUP_N},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_Q in [32, 64]
        for BLOCK_KV in [64, 128]
        for GROUP_N in GROUP_NM_SWEEP
        for num_stages in NUM_STAGES_SWEEP
        for num_warps in NUM_WARPS_SWEEP
    ],
    key=KEY_CACHE,
)
@triton.jit
def _attn_bwd_dk_dv_rope(
    Q, K, V, dO, dK, dV, M, D,
    # Q strides [B,H,S,D]
    sqb, sqh, sqs, sqd,
    # K strides [B,H,S,D]
    skb, skh, sks, skd,
    # V strides [B,H,S,D]
    svb, svh, svs, svd,
    # dO strides [B,H,S,D]
    sob, soh, sos, sod,
    # dK strides [B,H,S,D]
    s_dkb, s_dkh, s_dks, s_dkd,
    # dV strides [B,H,S,D]
    s_dvb, s_dvh, s_dvs, s_dvd,
    # RoPE tables [N,P] + (row, col) strides
    COSX, SINX, COSY, SINY,
    cosx_s, cosx_p,
    sinx_s, sinx_p,
    cosy_s, cosy_p,
    siny_s, siny_p,
    # Meta
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    softmax_scale: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DTYPE: tl.constexpr,
    GROUP_N: tl.constexpr,
    D2: tl.constexpr,        # HEAD_DIM // 2
    P: tl.constexpr,         # HEAD_DIM // 4
    HAS_CLS: tl.constexpr,
):
    tl.static_assert((HEAD_DIM % 4) == 0)

    # -------------------------
    # 0) program ids / (b,h) plane
    # -------------------------
    pid_kv = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // NUM_HEADS
    h = pid_bh % NUM_HEADS

    # flatten M, D as [B*H, S]
    off_bh_seq = (pid_bh * SEQ_LEN).to(tl.int32)
    M = M + off_bh_seq
    D = D + off_bh_seq

    # -------------------------
    # 1) swizzled KV tile id (same as non-RoPE kernel)
    # -------------------------
    num_tiles_kv = tl.cdiv(SEQ_LEN, BLOCK_KV)
    group_id     = pid_kv // GROUP_N
    group_start  = group_id * GROUP_N
    if group_start >= num_tiles_kv:
        return

    tiles_in_group = tl.minimum(GROUP_N, num_tiles_kv - group_start)
    kv_in_group    = pid_kv - group_start
    kv_eff         = kv_in_group % tiles_in_group
    rot            = pid_bh % tiles_in_group
    kv_tile_id     = group_start + ((kv_eff + rot) % tiles_in_group)

    start_kv = kv_tile_id * BLOCK_KV
    if start_kv >= SEQ_LEN:
        return

    # -------------------------
    # 2) base offsets & strides
    # -------------------------
    off_bh_q  = (b * sqb   + h * sqh  ).to(tl.int32)
    off_bh_k  = (b * skb   + h * skh  ).to(tl.int32)
    off_bh_v  = (b * svb   + h * svh  ).to(tl.int32)
    off_bh_do = (b * sob   + h * soh  ).to(tl.int32)
    off_bh_dk = (b * s_dkb + h * s_dkh).to(tl.int32)
    off_bh_dv = (b * s_dvb + h * s_dvh).to(tl.int32)

    sqs_i   = tl.full((1,), sqs,   tl.int32)
    sqd_i   = tl.full((1,), sqd,   tl.int32)
    sks_i   = tl.full((1,), sks,   tl.int32)
    skd_i   = tl.full((1,), skd,   tl.int32)
    svs_i   = tl.full((1,), svs,   tl.int32)
    svd_i   = tl.full((1,), svd,   tl.int32)
    sos_i   = tl.full((1,), sos,   tl.int32)
    sod_i   = tl.full((1,), sod,   tl.int32)
    s_dks_i = tl.full((1,), s_dks, tl.int32)
    s_dkd_i = tl.full((1,), s_dkd, tl.int32)

    cosx_s_i = tl.full((1,), cosx_s, tl.int32)
    cosx_p_i = tl.full((1,), cosx_p, tl.int32)
    sinx_s_i = tl.full((1,), sinx_s, tl.int32)
    sinx_p_i = tl.full((1,), sinx_p, tl.int32)
    cosy_s_i = tl.full((1,), cosy_s, tl.int32)
    cosy_p_i = tl.full((1,), cosy_p, tl.int32)
    siny_s_i = tl.full((1,), siny_s, tl.int32)
    siny_p_i = tl.full((1,), siny_p, tl.int32)

    pair_ix = tl.arange(0, P).to(tl.int32)   # [P]
    even    = 2 * pair_ix                   # 0,2,4,...
    odd     = 2 * pair_ix + 1               # 1,3,5,...
    D2_i    = tl.full((1,), D2, tl.int32)

    # -------------------------
    # 3) this KV tile: indices + V block
    # -------------------------
    cols_kv = tl.arange(0, BLOCK_KV)
    k_idx   = start_kv + cols_kv          # [BKV]
    kv_valid = k_idx < SEQ_LEN
    k_idx64  = k_idx.to(tl.int32)

    d_idx = tl.arange(0, HEAD_DIM).to(tl.int32)[None, :]  # [1,D]

    # V tile [BKV, D]
    base_V = V + off_bh_v
    V_ptrs = base_V + k_idx64[:, None] * svs_i + d_idx * svd_i
    V_block = tl.load(V_ptrs,mask=kv_valid[:, None],other=0.0,).to(DTYPE)  # [BKV, D]

    # -------------------------
    # 4) K pairs + K-side RoPE ⇒ K̂
    # -------------------------
    base_K   = K + off_bh_k
    k_idx64b = k_idx64[None, :]  # [1,BKV]

    krow_xe = even * skd_i
    krow_xo = odd  * skd_i
    krow_ye = (D2_i + even) * skd_i
    krow_yo = (D2_i + odd ) * skd_i

    Kxe = tl.load(base_K + krow_xe[:, None] + k_idx64b * sks_i,mask=kv_valid[None, :],other=0.0).to(DTYPE)# [P, BKV]
    Kxo = tl.load(base_K + krow_xo[:, None] + k_idx64b * sks_i,mask=kv_valid[None, :],other=0.0).to(DTYPE)
    Kye = tl.load(base_K + krow_ye[:, None] + k_idx64b * sks_i,mask=kv_valid[None, :],other=0.0).to(DTYPE)
    Kyo = tl.load(base_K + krow_yo[:, None] + k_idx64b * sks_i,mask=kv_valid[None, :],other=0.0).to(DTYPE)

    # RoPE positions for keys
    lin_k = k_idx - HAS_CLS
    lin_k = tl.maximum(lin_k, 0).to(tl.int32)
    is_cls_k = (HAS_CLS != 0) & (k_idx == 0)

    pair_ix_col = pair_ix[:, None]      # [P,1]
    lin_k_row   = lin_k[None, :]        # [1,BKV]

    cxk = pair_ix_col * cosx_p_i + lin_k_row * cosx_s_i
    sxk = pair_ix_col * sinx_p_i + lin_k_row * sinx_s_i
    cyk = pair_ix_col * cosy_p_i + lin_k_row * cosy_s_i
    syk = pair_ix_col * siny_p_i + lin_k_row * siny_s_i

    CX_k = tl.load(COSX + cxk, mask=kv_valid[None, :], other=0.0).to(DTYPE)
    SX_k = tl.load(SINX + sxk, mask=kv_valid[None, :], other=0.0).to(DTYPE)
    CY_k = tl.load(COSY + cyk, mask=kv_valid[None, :], other=0.0).to(DTYPE)
    SY_k = tl.load(SINY + syk, mask=kv_valid[None, :], other=0.0).to(DTYPE)

    is_cls_k_bc = is_cls_k[None, :]     # [1,BKV]

    Kx_e_r = tl.where(is_cls_k_bc, Kxe, Kxe * CX_k - Kxo * SX_k)
    Kx_o_r = tl.where(is_cls_k_bc, Kxo, Kxe * SX_k + Kxo * CX_k)
    Ky_e_r = tl.where(is_cls_k_bc, Kye, Kye * CY_k - Kyo * SY_k)
    Ky_o_r = tl.where(is_cls_k_bc, Kyo, Kye * SY_k + Kyo * CY_k)

    # -------------------------
    # 5) accumulators (V and K̂ pairs)
    # -------------------------
    dV_acc        = tl.zeros((BLOCK_KV, HEAD_DIM), dtype=tl.float32)
    dKx_e_hat_acc = tl.zeros((BLOCK_KV, P),        dtype=tl.float32)
    dKx_o_hat_acc = tl.zeros((BLOCK_KV, P),        dtype=tl.float32)
    dKy_e_hat_acc = tl.zeros((BLOCK_KV, P),        dtype=tl.float32)
    dKy_o_hat_acc = tl.zeros((BLOCK_KV, P),        dtype=tl.float32)

    s_fp32 = tl.full((1,), softmax_scale, dtype=tl.float32)

    # -------------------------
    # 6) loop over Q tiles
    # -------------------------
    num_q_tiles = tl.cdiv(SEQ_LEN, BLOCK_Q)
    base_Q  = Q  + off_bh_q
    base_dO = dO + off_bh_do

    for qi in range(0, num_q_tiles):
        start_q = qi * BLOCK_Q
        rows_q  = start_q + tl.arange(0, BLOCK_Q)
        q_valid = rows_q < SEQ_LEN
        rows64  = rows_q.to(tl.int32)

        mask_qk = q_valid[:, None] & kv_valid[None, :]

        # ---- dO tile [BQ, D] ----
        dO_ptrs = base_dO + rows64[:, None] * sos_i + d_idx * sod_i
        dO_block = tl.load(dO_ptrs,mask=q_valid[:, None],other=0.0).to(DTYPE)  # [BQ, D]

        # ---- Q pairs [BQ, P] ----
        row_off_q = rows64[:, None] * sqs_i

        qcol_xe = even * sqd_i
        qcol_xo = odd  * sqd_i
        qcol_ye = (D2_i + even) * sqd_i
        qcol_yo = (D2_i + odd ) * sqd_i

        Qxe = tl.load(base_Q + row_off_q + qcol_xe[None, :],mask=q_valid[:, None],other=0.0).to(DTYPE)
        Qxo = tl.load(base_Q + row_off_q + qcol_xo[None, :],mask=q_valid[:, None],other=0.0).to(DTYPE)
        Qye = tl.load(base_Q + row_off_q + qcol_ye[None, :],mask=q_valid[:, None],other=0.0).to(DTYPE)
        Qyo = tl.load(base_Q + row_off_q + qcol_yo[None, :],mask=q_valid[:, None],other=0.0).to(DTYPE)

        # ---- Q-side RoPE → Q̂ [BQ,P] ----
        lin_q = rows_q - HAS_CLS
        lin_q = tl.maximum(lin_q, 0).to(tl.int32)
        is_cls_q = (HAS_CLS != 0) & (rows_q == 0)
        is_cls_q_bc = is_cls_q[:, None]

        lin_q_col   = lin_q[:, None]      # [BQ,1]
        pair_ix_row = pair_ix[None, :]    # [1,P]

        cx_row = lin_q_col * cosx_s_i
        sx_row = lin_q_col * sinx_s_i
        cy_row = lin_q_col * cosy_s_i
        sy_row = lin_q_col * siny_s_i

        cx_col = pair_ix_row * cosx_p_i
        sx_col = pair_ix_row * sinx_p_i
        cy_col = pair_ix_row * cosy_p_i
        sy_col = pair_ix_row * siny_p_i

        CX_q = tl.load(COSX + cx_row + cx_col,mask=q_valid[:, None],other=0.0).to(DTYPE)
        SX_q = tl.load(SINX + sx_row + sx_col,mask=q_valid[:, None],other=0.0).to(DTYPE)
        CY_q = tl.load(COSY + cy_row + cy_col,mask=q_valid[:, None],other=0.0).to(DTYPE)
        SY_q = tl.load(SINY + sy_row + sy_col,mask=q_valid[:, None],other=0.0).to(DTYPE)

        Qx_e_r = tl.where(is_cls_q_bc, Qxe, Qxe * CX_q - Qxo * SX_q)
        Qx_o_r = tl.where(is_cls_q_bc, Qxo, Qxe * SX_q + Qxo * CX_q)
        Qy_e_r = tl.where(is_cls_q_bc, Qye, Qye * CY_q - Qyo * SY_q)
        Qy_o_r = tl.where(is_cls_q_bc, Qyo, Qye * SY_q + Qyo * CY_q)

        # ---- logits S (with RoPE) [BQ, BKV] ----
        S_blk = (tl.dot(Qx_e_r, Kx_e_r)
                 + tl.dot(Qx_o_r, Kx_o_r)
                 + tl.dot(Qy_e_r, Ky_e_r)
                 + tl.dot(Qy_o_r, Ky_o_r))
        S_blk = tl.where(mask_qk, S_blk * s_fp32, -float("inf"))

        # ---- reconstruct P from M (logsumexp) ----
        m_i = tl.load(M + rows64, mask=q_valid, other=0.0).to(tl.float32)  # [BQ]
        Di  = tl.load(D + rows64, mask=q_valid, other=0.0).to(tl.float32)  # [BQ]
        P_blk = tl.exp(S_blk - m_i[:, None])  # [BQ, BKV] (softmax probs)

        # ---- dV = Σ_i P_ij dO_i  => dV_acc [BKV,D] ----
        dV_acc = tl.dot(
            tl.trans(P_blk).to(DTYPE),   # [BKV,BQ]
            dO_block.to(DTYPE),          # [BQ,D]
            dV_acc                        # [BKV,D] (accumulator)
        )

        # ---- dS via FlashAttn formula ----
        dp_blk = tl.dot(dO_block, tl.trans(V_block)).to(tl.float32)
        dS_blk = (P_blk * (dp_blk - Di[:, None])).to(DTYPE)  # [BQ,BKV]

        # ---- dK̂ pairs: dK̂_j = Σ_i dS_ij Q̂_i ----
        dKx_e_hat_acc = tl.dot(tl.trans(dS_blk), Qx_e_r, dKx_e_hat_acc)
        dKx_o_hat_acc = tl.dot(tl.trans(dS_blk), Qx_o_r, dKx_o_hat_acc)
        dKy_e_hat_acc = tl.dot(tl.trans(dS_blk), Qy_e_r, dKy_e_hat_acc)
        dKy_o_hat_acc = tl.dot(tl.trans(dS_blk), Qy_o_r, dKy_o_hat_acc)

    # -------------------------
    # 7) apply softmax_scale to dK̂
    # -------------------------
    dKx_e_hat_acc *= s_fp32
    dKx_o_hat_acc *= s_fp32
    dKy_e_hat_acc *= s_fp32
    dKy_o_hat_acc *= s_fp32

    # -------------------------
    # 8) un-rotate K̂-grads → base dK pairs
    # -------------------------
    # CX_k, SX_k, CY_k, SY_k were built as [P, BKV] earlier.
    # For per-key rotation in pair-space, transpose to [BKV, P].
    CXk_T = tl.trans(CX_k).to(tl.float32)  # [BKV, P]
    SXk_T = tl.trans(SX_k).to(tl.float32)  # [BKV, P]
    CYk_T = tl.trans(CY_k).to(tl.float32)  # [BKV, P]
    SYk_T = tl.trans(SY_k).to(tl.float32)  # [BKV, P]

    # expand CLS mask to [BKV, 1] to broadcast across P
    is_cls_k_bc = is_cls_k[:, None]       # [BKV, 1]

    # For each key j, we apply R_j^T to its 2D pair (even, odd) components:
    # [d_e, d_o] = R^T * [dê, dô]
    dKx_e = tl.where(
        is_cls_k_bc,
        dKx_e_hat_acc,
        dKx_e_hat_acc * CXk_T + dKx_o_hat_acc * SXk_T,
    )
    dKx_o = tl.where(
        is_cls_k_bc,
        dKx_o_hat_acc,
        -dKx_e_hat_acc * SXk_T + dKx_o_hat_acc * CXk_T,
    )
    dKy_e = tl.where(
        is_cls_k_bc,
        dKy_e_hat_acc,
        dKy_e_hat_acc * CYk_T + dKy_o_hat_acc * SYk_T,
    )
    dKy_o = tl.where(
        is_cls_k_bc,
        dKy_o_hat_acc,
        -dKy_e_hat_acc * SYk_T + dKy_o_hat_acc * CYk_T,
    )

    # -------------------------
    # 9) scatter pairs into dK[:,D]
    # -------------------------
    base_dK = dK + off_bh_dk
    k_idx64_b = k_idx64[:, None] * s_dks_i  # [BKV,1]

    # x-even
    col_even = even[None, :] * s_dkd_i
    ptrs = base_dK + k_idx64_b + col_even
    tl.store(ptrs, dKx_e.to(dK.type.element_ty), mask=kv_valid[:, None])

    # x-odd
    col_odd = odd[None, :] * s_dkd_i
    ptrs = base_dK + k_idx64_b + col_odd
    tl.store(ptrs, dKx_o.to(dK.type.element_ty), mask=kv_valid[:, None])

    # y-even (offset by D2)
    col_even_y = (D2_i + even)[None, :] * s_dkd_i
    ptrs = base_dK + k_idx64_b + col_even_y
    tl.store(ptrs, dKy_e.to(dK.type.element_ty), mask=kv_valid[:, None])

    # y-odd (offset by D2)
    col_odd_y = (D2_i + odd)[None, :] * s_dkd_i
    ptrs = base_dK + k_idx64_b + col_odd_y
    tl.store(ptrs, dKy_o.to(dK.type.element_ty), mask=kv_valid[:, None])

    # -------------------------
    # 10) write dV block
    # -------------------------
    dV_blk = tl.make_block_ptr(
        dV + off_bh_dv,
        (SEQ_LEN, HEAD_DIM),
        (s_dvs, s_dvd),
        (start_kv, 0),
        (BLOCK_KV, HEAD_DIM),
        (1, 0),
    )
    tl.store(dV_blk, dV_acc.to(dV.type.element_ty), boundary_check=(0, 1))

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_Q": BLOCK_Q, "BLOCK_KV": BLOCK_KV, "GROUP_M": GROUP_M},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_Q in [64, 128]
        for BLOCK_KV in [32, 64]
        for GROUP_M in GROUP_NM_SWEEP
        for num_stages in NUM_STAGES_SWEEP
        for num_warps in NUM_WARPS_SWEEP
    ],
    key=KEY_CACHE,
)
@triton.jit
def _attn_bwd_dq_rope(
    Q, K, V, dO, dQ, M, D,          # tensors
    # Q strides [B,H,S,D]
    sqb, sqh, sqs, sqd,
    # K strides [B,H,S,D]
    skb, skh, sks, skd,
    # V strides [B,H,S,D]
    svb, svh, svs, svd,
    # dO strides [B,H,S,D]
    dob, doh, dos, dod,
    # dQ strides [B,H,S,D]
    s_dqb, s_dqh, s_dqs, s_dqd,
    # RoPE tables + (row, col) strides
    COSX, SINX, COSY, SINY,
    cosx_s, cosx_p,
    sinx_s, sinx_p,
    cosy_s, cosy_p,
    siny_s, siny_p,
    # meta
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DTYPE: tl.constexpr,
    GROUP_M: tl.constexpr,        
    BATCH_SIZE: tl.constexpr,
    softmax_scale: tl.constexpr,
    D2: tl.constexpr,
    P_pairs: tl.constexpr,        # HEAD_DIM // 4
    HAS_CLS: tl.constexpr,
):
    tl.static_assert((HEAD_DIM % 4) == 0)

    # -------------------------
    # 0) program ids
    # -------------------------
    pid_q  = tl.program_id(0)      # which Q tile
    pid_bh = tl.program_id(1)      # packed (b,h)

    b = pid_bh // NUM_HEADS
    h = pid_bh %  NUM_HEADS
    
    num_tiles_q = tl.cdiv(SEQ_LEN, BLOCK_Q)
    group_id    = pid_q // GROUP_M
    group_start = group_id * GROUP_M
    if group_start >= num_tiles_q:
        return
    
    tiles_in_grp = tl.minimum(GROUP_M, num_tiles_q - group_start)
    q_in_grp     = pid_q - group_start
    q_eff        = q_in_grp % tiles_in_grp
    rot          = pid_bh % tiles_in_grp
    q_tile_id    = group_start + ((q_eff + rot) % tiles_in_grp)

    # one tile in q-dim
    start_q = q_tile_id * BLOCK_Q
    if start_q >= SEQ_LEN:
        return

    # -------------------------
    # 1) base offsets & scalars
    # -------------------------
    off_bh_q  = (b * sqb   + h * sqh  ).to(tl.int32)
    off_bh_k  = (b * skb   + h * skh  ).to(tl.int32)
    off_bh_v  = (b * svb   + h * svh  ).to(tl.int32)
    off_bh_dO = (b * dob   + h * doh  ).to(tl.int32)
    off_bh_dQ = (b * s_dqb + h * s_dqh).to(tl.int32)

    # treat M, D as [B*H, S] slices
    off_bh_seq = (pid_bh * SEQ_LEN).to(tl.int32)
    M = M + off_bh_seq
    D = D + off_bh_seq

    # scalar strides
    sqs_i   = tl.full((1,), sqs,   tl.int32)
    sqd_i   = tl.full((1,), sqd,   tl.int32)
    sks_i   = tl.full((1,), sks,   tl.int32)
    skd_i   = tl.full((1,), skd,   tl.int32)
    svs_i   = tl.full((1,), svs,   tl.int32)
    svd_i   = tl.full((1,), svd,   tl.int32)
    s_dqs_i = tl.full((1,), s_dqs, tl.int32)
    s_dqd_i = tl.full((1,), s_dqd, tl.int32)

    cosx_s_i = tl.full((1,), cosx_s, tl.int32)
    cosx_p_i = tl.full((1,), cosx_p, tl.int32)
    sinx_s_i = tl.full((1,), sinx_s, tl.int32)
    sinx_p_i = tl.full((1,), sinx_p, tl.int32)
    cosy_s_i = tl.full((1,), cosy_s, tl.int32)
    cosy_p_i = tl.full((1,), cosy_p, tl.int32)
    siny_s_i = tl.full((1,), siny_s, tl.int32)
    siny_p_i = tl.full((1,), siny_p, tl.int32)

    # -------------------------
    # 2) Q rows, masks, pair indices
    # -------------------------
    rows    = start_q + tl.arange(0, BLOCK_Q)    # [BLOCK_Q]
    rows64  = rows.to(tl.int32)
    q_valid = rows < SEQ_LEN

    pair_ix = tl.arange(0, P_pairs).to(tl.int32)
    even    = 2 * pair_ix
    odd     = 2 * pair_ix + 1
    D2_i    = tl.full((1,), D2, tl.int32)

    d_idx = tl.arange(0, HEAD_DIM).to(tl.int32)[None, :]  # [1, D]

    # -------------------------
    # 3) load dO tile [BQ, D] and rowwise D, M
    # -------------------------
    dO_ptrs  = (dO + off_bh_dO) + rows64[:, None] * dos + d_idx * dod
    dO_block = tl.load(dO_ptrs,mask=q_valid[:, None],other=0.0).to(DTYPE)  # [BQ, D]

    Di = tl.load(D + rows64, mask=q_valid, other=0.0).to(tl.float32)         # [BQ]
    Mi = tl.load(M + rows64, mask=q_valid, other=-float("inf")).to(tl.float32)  # [BQ]

    # -------------------------
    # 4) gather Q pairs & apply RoPE (same as fwd)
    # -------------------------
    base_Q    = Q + off_bh_q
    row_off_q = rows64[:, None] * sqs_i

    qcol_xe = even * sqd_i
    qcol_xo = odd  * sqd_i
    qcol_ye = (D2_i + even) * sqd_i
    qcol_yo = (D2_i + odd ) * sqd_i

    Qxe = tl.load(base_Q + row_off_q + qcol_xe[None, :],mask=q_valid[:, None],other=0.0).to(DTYPE)  # [BQ, P_pairs]
    Qxo = tl.load(base_Q + row_off_q + qcol_xo[None, :],mask=q_valid[:, None],other=0.0).to(DTYPE)
    Qye = tl.load(base_Q + row_off_q + qcol_ye[None, :],mask=q_valid[:, None],other=0.0).to(DTYPE)
    Qyo = tl.load(base_Q + row_off_q + qcol_yo[None, :],mask=q_valid[:, None],other=0.0).to(DTYPE)

    # Q-side RoPE
    lin_q    = rows - HAS_CLS
    lin_q    = tl.maximum(lin_q, 0).to(tl.int32)
    is_cls_q = (HAS_CLS != 0) & (rows == 0)

    cx_row_q = lin_q[:, None] * cosx_s_i
    sx_row_q = lin_q[:, None] * sinx_s_i
    cy_row_q = lin_q[:, None] * cosy_s_i
    sy_row_q = lin_q[:, None] * siny_s_i

    cx_col_q = pair_ix[None, :] * cosx_p_i
    sx_col_q = pair_ix[None, :] * sinx_p_i
    cy_col_q = pair_ix[None, :] * cosy_p_i
    sy_col_q = pair_ix[None, :] * siny_p_i

    CX_q = tl.load(COSX + cx_row_q + cx_col_q,mask=q_valid[:, None],other=0.0).to(DTYPE)
    SX_q = tl.load(SINX + sx_row_q + sx_col_q,mask=q_valid[:, None],other=0.0).to(DTYPE)
    CY_q = tl.load(COSY + cy_row_q + cy_col_q,mask=q_valid[:, None],other=0.0).to(DTYPE)
    SY_q = tl.load(SINY + sy_row_q + sy_col_q,mask=q_valid[:, None],other=0.0).to(DTYPE)

    is_cls_q_bc = is_cls_q[:, None]

    # rotated Q̂ pairs (same as fwd)
    Qx_e_r = tl.where(is_cls_q_bc, Qxe, Qxe * CX_q - Qxo * SX_q)
    Qx_o_r = tl.where(is_cls_q_bc, Qxo, Qxe * SX_q + Qxo * CX_q)
    Qy_e_r = tl.where(is_cls_q_bc, Qye, Qye * CY_q - Qyo * SY_q)
    Qy_o_r = tl.where(is_cls_q_bc, Qyo, Qye * SY_q + Qyo * CY_q)

    # -------------------------
    # 5) accumulators in rotated Q basis
    # -------------------------
    dQx_e_hat_acc = tl.zeros((BLOCK_Q, P_pairs), dtype=tl.float32)
    dQx_o_hat_acc = tl.zeros((BLOCK_Q, P_pairs), dtype=tl.float32)
    dQy_e_hat_acc = tl.zeros((BLOCK_Q, P_pairs), dtype=tl.float32)
    dQy_o_hat_acc = tl.zeros((BLOCK_Q, P_pairs), dtype=tl.float32)

    # -------------------------
    # 6) loop over KV tiles: rebuild S->P, then dS->dQ̂
    # -------------------------
    num_kv_tiles = tl.cdiv(SEQ_LEN, BLOCK_KV)
    cols_local   = tl.arange(0, BLOCK_KV)

    for kv_t in range(0, num_kv_tiles):
        start_kv = kv_t * BLOCK_KV
        k_idx    = start_kv + cols_local           # [BLOCK_KV]
        kv_valid = k_idx < SEQ_LEN
        k_idx64  = k_idx.to(tl.int32)

        # V tile [BKV, D] (for dP)
        V_ptrs = (V + off_bh_v) + k_idx64[:, None] * svs + d_idx * svd
        V_blk  = tl.load(V_ptrs,mask=kv_valid[:, None],other=0.0,).to(DTYPE)  # [BKV, D]

        # K pairs [P_pairs, BKV]
        base_K   = K + off_bh_k
        kv_colsK = k_idx64[None, :] * sks_i

        krow_xe = even * skd_i
        krow_xo = odd  * skd_i
        krow_ye = (D2_i + even) * skd_i
        krow_yo = (D2_i + odd ) * skd_i

        Kxe = tl.load(base_K + krow_xe[:, None] + kv_colsK,mask=kv_valid[None, :],other=0.0).to(DTYPE)
        Kxo = tl.load(base_K + krow_xo[:, None] + kv_colsK,mask=kv_valid[None, :],other=0.0).to(DTYPE)
        Kye = tl.load(base_K + krow_ye[:, None] + kv_colsK,mask=kv_valid[None, :],other=0.0).to(DTYPE)
        Kyo = tl.load(base_K + krow_yo[:, None] + kv_colsK,mask=kv_valid[None, :],other=0.0).to(DTYPE)

        # K-side RoPE
        lin_k    = k_idx - HAS_CLS
        lin_k    = tl.maximum(lin_k, 0).to(tl.int32)
        is_cls_k = (HAS_CLS != 0) & (k_idx == 0)

        cxk = pair_ix[:, None] * cosx_p_i + lin_k[None, :] * cosx_s_i
        sxk = pair_ix[:, None] * sinx_p_i + lin_k[None, :] * sinx_s_i
        cyk = pair_ix[:, None] * cosy_p_i + lin_k[None, :] * cosy_s_i
        syk = pair_ix[:, None] * siny_p_i + lin_k[None, :] * siny_s_i

        CX_k = tl.load(COSX + cxk,mask=kv_valid[None, :],other=0.0).to(DTYPE)
        SX_k = tl.load(SINX + sxk,mask=kv_valid[None, :],other=0.0).to(DTYPE)
        CY_k = tl.load(COSY + cyk,mask=kv_valid[None, :],other=0.0).to(DTYPE)
        SY_k = tl.load(SINY + syk,mask=kv_valid[None, :],other=0.0).to(DTYPE)

        is_cls_k_bc = is_cls_k[None, :]

        Kx_e_r = tl.where(is_cls_k_bc, Kxe, Kxe * CX_k - Kxo * SX_k)
        Kx_o_r = tl.where(is_cls_k_bc, Kxo, Kxe * SX_k + Kxo * CX_k)
        Ky_e_r = tl.where(is_cls_k_bc, Kye, Kye * CY_k - Kyo * SY_k)
        Ky_o_r = tl.where(is_cls_k_bc, Kyo, Kye * SY_k + Kyo * CY_k)

        # ---- rebuild logits S_tile (same as fwd) ----
        S_tile = (tl.dot(Qx_e_r, Kx_e_r) 
               + tl.dot(Qx_o_r, Kx_o_r) 
               + tl.dot(Qy_e_r, Ky_e_r) 
               + tl.dot(Qy_o_r, Ky_o_r)).to(tl.float32)

        S_tile = S_tile * tl.full((1,), softmax_scale, dtype=tl.float32)
        S_tile = tl.where(q_valid[:, None] & kv_valid[None, :],S_tile,-float("inf"))

        # ---- reconstruct P from S and M: P = exp(S - M) ----
        P_blk = tl.exp(S_tile - Mi[:, None])             # [BQ,BKV]
        P_blk = tl.where(q_valid[:, None] & kv_valid[None, :],P_blk,0.0).to(DTYPE)

        # ---- dP = dO @ Vᵀ ----
        dP_blk = tl.dot(dO_block.to(tl.float32), tl.trans(V_blk).to(tl.float32))  # [BQ,BKV]

        # ---- dS = P * (dP - D) ----
        dS_blk = (P_blk * (dP_blk - Di[:, None])).to(DTYPE)  # [BQ,BKV]

        # ---- accumulate dQ̂ in pair-space: [BQ,P_pairs] ----
        dQx_e_hat_acc = tl.dot(dS_blk,tl.trans(Kx_e_r), dQx_e_hat_acc)
        dQx_o_hat_acc = tl.dot(dS_blk,tl.trans(Kx_o_r), dQx_o_hat_acc)
        dQy_e_hat_acc = tl.dot(dS_blk,tl.trans(Ky_e_r), dQy_e_hat_acc)
        dQy_o_hat_acc = tl.dot(dS_blk,tl.trans(Ky_o_r), dQy_o_hat_acc)

    # -------------------------
    # 7) apply softmax_scale once (chain rule for scaled logits)
    # -------------------------
    s_fp32 = tl.full((1,), softmax_scale, dtype=tl.float32)
    dQx_e_hat_acc *= s_fp32
    dQx_o_hat_acc *= s_fp32
    dQy_e_hat_acc *= s_fp32
    dQy_o_hat_acc *= s_fp32

    # -------------------------
    # 8) un-rotate dQ̂ → dQ (Rᵀ in Q space)
    # -------------------------
    CX_q_f = CX_q.to(tl.float32)
    SX_q_f = SX_q.to(tl.float32)
    CY_q_f = CY_q.to(tl.float32)
    SY_q_f = SY_q.to(tl.float32)
    is_cls_q_bc_f = is_cls_q_bc

    dQx_e = tl.where(
        is_cls_q_bc_f,
        dQx_e_hat_acc,
        dQx_e_hat_acc * CX_q_f + dQx_o_hat_acc * SX_q_f,
    )
    dQx_o = tl.where(
        is_cls_q_bc_f,
        dQx_o_hat_acc,
        -dQx_e_hat_acc * SX_q_f + dQx_o_hat_acc * CX_q_f,
    )
    dQy_e = tl.where(
        is_cls_q_bc_f,
        dQy_e_hat_acc,
        dQy_e_hat_acc * CY_q_f + dQy_o_hat_acc * SY_q_f,
    )
    dQy_o = tl.where(
        is_cls_q_bc_f,
        dQy_o_hat_acc,
        -dQy_e_hat_acc * SY_q_f + dQy_o_hat_acc * CY_q_f,
    )

    # -------------------------
    # 9) scatter (x/y, even/odd) pairs into dQ[:, D]
    # -------------------------
    base_dQ = dQ + off_bh_dQ
    row_ix  = rows64[:, None] * s_dqs_i  # [BQ,1]

    # x-even
    col_ix = even[None, :] * s_dqd_i
    ptrs   = base_dQ + row_ix + col_ix
    tl.store(ptrs,dQx_e.to(dQ.type.element_ty),mask=q_valid[:, None])

    # x-odd
    col_ix = odd[None, :] * s_dqd_i
    ptrs   = base_dQ + row_ix + col_ix
    tl.store(ptrs,dQx_o.to(dQ.type.element_ty),mask=q_valid[:, None])

    # y-even (offset by D2)
    col_ix = (D2_i + even)[None, :] * s_dqd_i
    ptrs   = base_dQ + row_ix + col_ix
    tl.store(ptrs,dQy_e.to(dQ.type.element_ty),mask=q_valid[:, None])

    # y-odd (offset by D2)
    col_ix = (D2_i + odd)[None, :] * s_dqd_i
    ptrs   = base_dQ + row_ix + col_ix
    tl.store(ptrs,dQy_o.to(dQ.type.element_ty),mask=q_valid[:, None])

class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, cos_sin, H_img=14, has_cls=True):
        # ---- Shapes / dtypes ----
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.size()
        assert (HEAD_DIM % 4) == 0, "HEAD_DIM must be divisible by 4 (2D RoPE requires pairs)."
        N_img = H_img ** 2
        if has_cls:
            assert SEQ_LEN == 1 + N_img, f"SEQ_LEN must be 1 + H_img*W_img when has_cls=True (got {SEQ_LEN} vs {1+N_img})."
        else:
            assert SEQ_LEN == N_img, f"SEQ_LEN must equal H_img*W_img when has_cls=False (got {SEQ_LEN} vs {N_img})."

        
        print("[TRITON WRAPPER] Q/K/V")
        print("  Q.shape:", Q.shape, "Q.stride:", Q.stride())
        print("  K.shape:", K.shape, "K.stride:", K.stride())
        print("  V.shape:", V.shape, "V.stride:", V.stride())
    
        comp_triton = _sdpa_comp_dtype(Q)
        softmax_scale = 1.0 / (HEAD_DIM ** 0.5)

        # ---- Outputs ----
        O = torch.empty(Q.shape, dtype=Q.dtype, device=Q.device)
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        ) 
        O_debug = torch.randn(Q.shape, dtype=Q.dtype, device=Q.device)

        # ---- RoPE tables [N, P] (float32) ----
        COSX, SINX, COSY, SINY = cos_sin.tables()
        P_pairs = HEAD_DIM // 4

        # ---- Launch ----
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_Q"]),
            BATCH_SIZE * NUM_HEADS,
        )
        _attn_fwd[grid](
            Q, K, V, M, O,
            # strides: Q,K,V,O
            *Q.stride(), *K.stride(), *V.stride(), *O.stride(),
            # RoPE tables + their (row, col) strides
            COSX, SINX, COSY, SINY,
            COSX.stride(0), COSX.stride(1),
            SINX.stride(0), SINX.stride(1),
            COSY.stride(0), COSY.stride(1),
            SINY.stride(0), SINY.stride(1),
            # meta
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            D2=HEAD_DIM // 2,
            P=P_pairs,
            BATCH_SIZE=BATCH_SIZE,
            HAS_CLS=int(has_cls),
            softmax_scale=softmax_scale,
            DTYPE=comp_triton,
        )

        # ---- Save for backward ----
        ctx.softmax_scale = softmax_scale
        ctx.comp_triton = comp_triton
        ctx.has_cls = has_cls
        ctx.save_for_backward(Q, K, V, O_debug, M, COSX, SINX, COSY, SINY)
        return O_debug
    

    @staticmethod
    def backward(ctx, dO):
        print("[TRITON WRAPPER] Q/K/V")
        print("  dO.shape:", dO.shape, "dO.stride:", dO.stride())
        Q, K, V, O, M, COSX, SINX, COSY, SINY = ctx.saved_tensors
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.size()
        
        dQ = torch.empty(Q.shape, dtype=Q.dtype, device=Q.device) 
        dK = torch.empty(K.shape, dtype=K.dtype, device=K.device)
        dV = torch.empty(V.shape, dtype=V.dtype, device=V.device)
        D = torch.empty(M.shape, dtype=M.dtype, device=M.device) 
        pre_grid = lambda meta: (triton.cdiv(SEQ_LEN, meta["BLOCK_Q"]),
                         BATCH_SIZE * NUM_HEADS)
        _attn_bwd_preprocess[pre_grid](
            O, dO, D, *O.stride(), *dO.stride(),
            NUM_HEADS=NUM_HEADS, SEQ_LEN=SEQ_LEN, HEAD_DIM=HEAD_DIM,
            BATCH_SIZE=BATCH_SIZE,
        )
        
        dkdv_grid = lambda meta: (
            triton.cdiv(SEQ_LEN, meta["BLOCK_KV"]),
            BATCH_SIZE * NUM_HEADS,
        )

        _attn_bwd_dk_dv_rope[dkdv_grid](
            Q, K, V, dO,
            dK, dV,
            M, D,
            # strides
            *Q.stride(), *K.stride(), *V.stride(), *dO.stride(),
            *dK.stride(), *dV.stride(),
            # RoPE tables + strides
            COSX, SINX, COSY, SINY,
            COSX.stride(0), COSX.stride(1),
            SINX.stride(0), SINX.stride(1),
            COSY.stride(0), COSY.stride(1),
            SINY.stride(0), SINY.stride(1),
            # meta
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            softmax_scale=ctx.softmax_scale,
            HEAD_DIM=HEAD_DIM,
            BATCH_SIZE=BATCH_SIZE,
            DTYPE=ctx.comp_triton,
            D2=HEAD_DIM // 2,
            P=HEAD_DIM // 4,
            HAS_CLS=int(ctx.has_cls),
        )
        
        dq_grid = lambda meta: (
            triton.cdiv(SEQ_LEN, meta["BLOCK_Q"]),
            BATCH_SIZE * NUM_HEADS,
        )

        _attn_bwd_dq_rope[dq_grid](
            # tensors
            Q, K, V, dO, dQ, M, D,
            # strides Q,K,V,dO,dQ
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *dO.stride(),
            *dQ.stride(),
            # RoPE tables + strides
            COSX, SINX, COSY, SINY,
            COSX.stride(0), COSX.stride(1),
            SINX.stride(0), SINX.stride(1),
            COSY.stride(0), COSY.stride(1),
            SINY.stride(0), SINY.stride(1),
            # meta
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BATCH_SIZE=BATCH_SIZE,
            HEAD_DIM=HEAD_DIM,
            DTYPE=ctx.comp_triton,
            softmax_scale=ctx.softmax_scale,
            D2=HEAD_DIM // 2,
            P_pairs=HEAD_DIM // 4,
            HAS_CLS=int(ctx.has_cls),
        )
        
        return dQ, dK, dV, None, None, None, None

class CosSinTable(torch.nn.Module):
    def __init__(self, base, H_img=14, D=64, device="cuda"):
        super().__init__()
        COSX, SINX, COSY, SINY = self._rope_pairs_tables(base, H_img, D, device)

        # Register as buffers so they move with the module and go in state_dict
        self.register_buffer("COSX", COSX)
        self.register_buffer("SINX", SINX)
        self.register_buffer("COSY", COSY)
        self.register_buffer("SINY", SINY)

    def tables(self):
        return self.COSX, self.SINX, self.COSY, self.SINY

    def _rope_pairs_tables(self, base, H_img, D, device="cuda"):
        cos_x, sin_x, cos_y, sin_y = _build_axial_rope(
            H_img, D, device, base=base
        )
        COSX = cos_x[:, 0::2].contiguous()
        SINX = sin_x[:, 0::2].contiguous()
        COSY = cos_y[:, 0::2].contiguous()
        SINY = sin_y[:, 0::2].contiguous()
        return COSX, SINX, COSY, SINY
    
    
def sdpa_triton_fa_rope(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, cos_sin: CosSinTable):
    """
    Triton Scaled Dot-Product Attention (SDPA) with 2D Axial RoPE.

    Q, K, V: [B, H, S, D] (contiguous)
    S must be 1 + H_img*W_img if has_cls=True, else S == H_img*W_img.
    """
    #return TritonAttention.apply(Q.contiguous(), K.contiguous(), V.contiguous(), cos_sin)
    return TritonAttention.apply(Q, K, V, cos_sin)

