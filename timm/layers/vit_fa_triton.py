
import torch
from torch import nn
from torch import Tensor
import triton
import triton.language as tl

GROUP_NM_SWEEP = [2, 4, 8]
NUM_STAGES_SWEEP = [3, 4, 7]
NUM_WARPS_SWEEP = [2, 4]
KEY_CACHE = ["BATCH_SIZE", "NUM_HEADS", "SEQ_LEN", "HEAD_DIM"]

def _sdpa_comp_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else x.dtype

def _triton_compute_dtype(dtype: torch.dtype):
    if dtype is torch.float16:
        return tl.float16
    if dtype is torch.bfloat16:
        return tl.bfloat16
    if dtype is torch.float32:
        return tl.float32
    raise ValueError(f"Unsupported compute dtype for Triton SDPA: {dtype}")

@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    DTYPE: tl.constexpr,
):
    s = tl.full([1], softmax_scale, dtype=DTYPE)
    Q_block = Q_block * s
    for start_kv in range(0, SEQ_LEN, BLOCK_KV):
        K_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        S = tl.dot(Q_block, K_block) 

        kv_idx  = start_kv + offs_kv
        kv_valid = kv_idx < SEQ_LEN
        S = tl.where(kv_valid[None, :], S, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(S, axis=1))
        P_block = tl.exp(S - m_ij[:, None])
        l_ij = tl.sum(P_block, axis=1)

        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        P_block = P_block.to(DTYPE)
        V_block = V_block.to(DTYPE)

        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_KV))
    
    O_block = O_block / l_i[:, None]
    return O_block, l_i, m_i

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
    Q, K, V, softmax_scale, M, O,
    stride_batch, stride_head, stride_seq, stride_dim,
    BATCH_SIZE, NUM_HEADS: tl.constexpr, SEQ_LEN: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, DTYPE: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tl.static_assert(BLOCK_KV <= HEAD_DIM)

    # --- program ids ---
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    num_tiles_m   = tl.cdiv(SEQ_LEN, BLOCK_Q)                       # ceil_div
    group_id      = pid_m // GROUP_M
    tiles_in_this = tl.minimum(GROUP_M, num_tiles_m - group_id*GROUP_M)

    m_in_grp      = pid_m - group_id*GROUP_M                        # 0..GROUP_M-1
    m_in_grp_eff  = m_in_grp % tiles_in_this                        # clamp to tail size
    rot           = pid_bh % tiles_in_this
    m_swizzled    = group_id*GROUP_M + ((m_in_grp_eff + rot) % tiles_in_this)

    start_q       = m_swizzled * BLOCK_Q
    if start_q >= SEQ_LEN:
        return

    index_batch = pid_bh // NUM_HEADS
    index_head  = pid_bh %  NUM_HEADS

    off_bh = (index_batch.to(tl.int64) * stride_batch
              + index_head.to(tl.int64) * stride_head)
    
    # --- block pointers ---
    Q_block_ptr = tl.make_block_ptr(
        Q + off_bh, (SEQ_LEN, HEAD_DIM), (stride_seq, stride_dim), (start_q, 0), (BLOCK_Q, HEAD_DIM), (1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        V + off_bh, (SEQ_LEN, HEAD_DIM), (stride_seq, stride_dim), (0, 0), (BLOCK_KV, HEAD_DIM), (1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        K + off_bh, (HEAD_DIM, SEQ_LEN), (stride_dim, stride_seq), (0, 0), (HEAD_DIM, BLOCK_KV), (0, 1)
    )
    O_block_ptr = tl.make_block_ptr(
        O + off_bh, (SEQ_LEN, HEAD_DIM), (stride_seq, stride_dim), (start_q, 0), (BLOCK_Q, HEAD_DIM), (1, 0)
    )

    offs_q  = start_q + tl.arange(0, BLOCK_Q)
    offs_kv = tl.arange(0, BLOCK_KV)

    # --- per-row running stats + output tile ---
    m_i = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_Q,),  1.0,          dtype=tl.float32)
    O_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    Q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # --- inner loop over KV tiles (online softmax) ---
    O_block, l_i, m_i = _attn_fwd_inner(
        O_block, l_i, m_i, Q_block,
        K_block_ptr, V_block_ptr, m_swizzled, softmax_scale,
        BLOCK_Q, BLOCK_KV, offs_q, offs_kv, SEQ_LEN, DTYPE
    )

    # --- write back: store log-sum-exp (for bwd) and O ---
    m_i += tl.math.log(l_i)
    m_ptrs = M + pid_bh * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i, mask=offs_q < SEQ_LEN)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty), boundary_check=(0, 1))

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_Q": BLOCK_Q},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_Q in [32, 64, 128]
        for num_stages in NUM_STAGES_SWEEP
        for num_warps in NUM_WARPS_SWEEP
    ],
    key=KEY_CACHE,
)
@triton.jit
def _attn_bwd_preprocess(
    O, dO, D, SEQ_LEN,
    BLOCK_Q: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    block_index_q = tl.program_id(0)
    start_q = block_index_q * BLOCK_Q
    if start_q >= SEQ_LEN:
        return

    offs_q  = start_q + tl.arange(0, BLOCK_Q)
    index_batch_head = tl.program_id(1)
    offs_dim = tl.arange(0, HEAD_DIM)
    mask_qd = offs_q[:, None] < SEQ_LEN

    O_block = tl.load(
        O + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM + offs_dim[None, :],
        mask=mask_qd, other=0
    )
    dO_block = tl.load(
        dO + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM + offs_dim[None, :],
        mask=mask_qd, other=0
    ).to(tl.float32)

    D_block = tl.sum(dO_block * O_block, axis=1)
    tl.store(D + index_batch_head * SEQ_LEN + offs_q, D_block, mask=offs_q < SEQ_LEN)


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
def _attn_bwd_dk_dv(
    Q, K, V, softmax_scale, dO, dQ, dK, dV, M, D,
    stride_batch, stride_head, stride_seq, stride_dim,
    NUM_HEADS, SEQ_LEN,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, 
    HEAD_DIM: tl.constexpr, DTYPE: tl.constexpr, GROUP_N: tl.constexpr
):
    # --- program ids ---
    pid_kv = tl.program_id(0)                 # which KV block
    pid_bh = tl.program_id(1)                 # packed (batch, head)
    b = pid_bh // NUM_HEADS
    h = pid_bh %  NUM_HEADS

    # --- base offsets for this (batch, head) slice ---
    off_bh     = (stride_batch * b + stride_head * h).to(tl.int64)
    off_bh_seq = (pid_bh * SEQ_LEN).to(tl.int64)
    M  += off_bh_seq
    D  += off_bh_seq

    num_tiles_kv = tl.cdiv(SEQ_LEN, BLOCK_KV)
    group_id     = pid_kv // GROUP_N
    group_start  = group_id * GROUP_N
    if group_start >= num_tiles_kv:
        return
    tiles_in_this = tl.minimum(GROUP_N, num_tiles_kv - group_start)
    kv_in_grp     = pid_kv - group_start
    kv_eff        = kv_in_grp % tiles_in_this
    rot           = pid_bh % tiles_in_this
    kv_tile_id    = group_start + ((kv_eff + rot) % tiles_in_this)

    start_kv = kv_tile_id * BLOCK_KV
    if start_kv >= SEQ_LEN:
        return

    offs_kv  = start_kv + tl.arange(0, BLOCK_KV)  # absolute KV indices for this tile
    K_blk = tl.make_block_ptr( 
        K + off_bh, (HEAD_DIM, SEQ_LEN), (stride_dim, stride_seq), (0, start_kv), (HEAD_DIM, BLOCK_KV), (0, 1)
    ) # base,        shape,               strides,                  offsets,       block_shape,          order
    V_blk = tl.make_block_ptr( 
        V + off_bh,(SEQ_LEN, HEAD_DIM),(stride_seq, stride_dim),(start_kv, 0),(BLOCK_KV, HEAD_DIM),(1, 0)
    )
    dK_blk = tl.make_block_ptr( 
        dK + off_bh, (SEQ_LEN, HEAD_DIM), (stride_seq, stride_dim), (start_kv, 0), (BLOCK_KV, HEAD_DIM), (1, 0)
    )
    dV_blk = tl.make_block_ptr( 
        dV + off_bh,(SEQ_LEN, HEAD_DIM),(stride_seq, stride_dim),(start_kv, 0),(BLOCK_KV, HEAD_DIM),(1, 0)
    )
    qT_blk = tl.make_block_ptr( 
        Q + off_bh,(HEAD_DIM, SEQ_LEN),(stride_dim, stride_seq),(0, 0),(HEAD_DIM, BLOCK_Q),(0, 1)
    )
    dO_blk = tl.make_block_ptr( 
        dO + off_bh,(SEQ_LEN, HEAD_DIM),(stride_seq, stride_dim),(0, 0),(BLOCK_Q, HEAD_DIM),(1, 0)
    )

    dV_acc = tl.zeros((BLOCK_KV, HEAD_DIM), dtype=tl.float32)
    dK_acc = tl.zeros((BLOCK_KV, HEAD_DIM), dtype=tl.float32)
    s = tl.full([1], softmax_scale, dtype=DTYPE)
    K_block = tl.load(K_blk, boundary_check=(0, 1), padding_option="zero") * s
    V_block = tl.load(V_blk, boundary_check=(0, 1), padding_option="zero")
    
    # Loop over Q tiles
    num_steps = tl.cdiv(SEQ_LEN, BLOCK_Q)
    for qi in range(0, num_steps):
        qT_block = tl.load(qT_blk, boundary_check=(0, 1), padding_option="zero")
        dO_block = tl.load(dO_blk, boundary_check=(0, 1), padding_option="zero")
        start_q = qi * BLOCK_Q
        offs_q  = start_q + tl.arange(0, BLOCK_Q)

        m  = tl.load(M + offs_q, mask=offs_q < SEQ_LEN, other=0.0)
        Di = tl.load(D + offs_q, mask=offs_q < SEQ_LEN, other=0.0)

        kv_valid = offs_kv < SEQ_LEN
        QK_T = tl.dot(tl.trans(K_block), qT_block) 
        QK_T = tl.where(kv_valid[:, None], QK_T, -float("inf"))
        P_T = tl.exp(QK_T - m[None, :])

        # --- dV += Pᵀ @ dO  (match operand dtypes) ---
        dV_acc += tl.dot(P_T.to(DTYPE), dO_block.to(DTYPE))

        # --- dpᵀ = V @ dOᵀ, then dSᵀ = Pᵀ * (dpᵀ - Di) ---
        dpT = tl.dot(V_block.to(DTYPE), tl.trans(dO_block.to(DTYPE))).to(tl.float32)
        dS_T = (P_T * (dpT - Di[None, :])).to(DTYPE)
        dK_acc = tl.dot(dS_T, tl.trans(qT_block.to(DTYPE)), dK_acc)

        qT_blk = tl.advance(qT_blk, (0, BLOCK_Q))
        dO_blk = tl.advance(dO_blk, (BLOCK_Q, 0))

    # Tail-safe stores
    dK_acc *= s 
    tl.store(dV_blk, dV_acc.to(dV.type.element_ty), boundary_check=(0, 1))
    tl.store(dK_blk, dK_acc.to(dK.type.element_ty), boundary_check=(0, 1))
    

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_Q": BLOCK_Q, "BLOCK_KV": BLOCK_KV, "GROUP_N": GROUP_N},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_Q in [64, 128]
        for BLOCK_KV in [32, 64]
        for GROUP_N in GROUP_NM_SWEEP
        for num_stages in NUM_STAGES_SWEEP
        for num_warps in NUM_WARPS_SWEEP
    ],
    key=KEY_CACHE,
)
@triton.jit
def _attn_bwd_dq(
    Q, K, V, softmax_scale, dO, dQ, dK, dV, M, D,
    stride_batch, stride_head, stride_seq, stride_dim,
    NUM_HEADS, SEQ_LEN,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, 
    HEAD_DIM: tl.constexpr, DTYPE: tl.constexpr,
    GROUP_N: tl.constexpr,                       # <-- added
):
    pid_bh = tl.program_id(1)
    b = pid_bh // NUM_HEADS
    h = pid_bh %  NUM_HEADS
    
    off_bh = (stride_batch * b + stride_head * h).to(tl.int64)
    off_bh_seq = (pid_bh * SEQ_LEN).to(tl.int64)
    M += off_bh_seq
    D += off_bh_seq

    # --- GROUP_M swizzle over Q tiles (tail-safe) ---
    pid_q = tl.program_id(0)
    num_tiles_m   = tl.cdiv(SEQ_LEN, BLOCK_Q)
    group_id      = pid_q // GROUP_N
    group_start   = group_id * GROUP_N
    # if this CTA's group starts past the last tile, exit early
    if group_start >= num_tiles_m:
        return
    tiles_in_this = tl.minimum(GROUP_N, num_tiles_m - group_start)
    m_in_grp      = pid_q - group_start
    m_eff         = m_in_grp % tiles_in_this
    rot           = pid_bh % tiles_in_this
    m_swizzled    = group_start + ((m_eff + rot) % tiles_in_this)

    start_q = m_swizzled * BLOCK_Q
    if start_q >= SEQ_LEN:
        return
    
    # ---------- block pointers ----------
    Q_blk = tl.make_block_ptr(
        Q + off_bh,(SEQ_LEN, HEAD_DIM),(stride_seq, stride_dim),(start_q, 0),(BLOCK_Q, HEAD_DIM),(1, 0),
    )
    dO_blk = tl.make_block_ptr(
        dO + off_bh,(SEQ_LEN, HEAD_DIM),(stride_seq, stride_dim),(start_q, 0),(BLOCK_Q, HEAD_DIM),(1, 0),
    )
    K_T_blk = tl.make_block_ptr(
        K + off_bh,(HEAD_DIM, SEQ_LEN),(stride_dim, stride_seq),(0, 0),(HEAD_DIM, BLOCK_KV),(0, 1),
    )
    V_T_blk = tl.make_block_ptr(
        V + off_bh,(HEAD_DIM, SEQ_LEN),(stride_dim, stride_seq),(0, 0),(HEAD_DIM, BLOCK_KV),(0, 1),
    )
    dQ_blk = tl.make_block_ptr(
        dQ + off_bh,(SEQ_LEN, HEAD_DIM),(stride_seq, stride_dim),(start_q, 0),(BLOCK_Q, HEAD_DIM),(1, 0),
    )

    # ---------- indices & constants ----------
    offs_q = start_q + tl.arange(0, BLOCK_Q)
    offs_kv = tl.arange(0, BLOCK_KV)

    # row-wise scalars
    m  = tl.load(M + offs_q, mask=offs_q < SEQ_LEN, other=0.0)[:, None]  # [BLOCK_Q, 1]
    Di = tl.load(D + offs_q, mask=offs_q < SEQ_LEN, other=0.0)           # [BLOCK_Q]
    s = tl.full([1], softmax_scale, dtype=DTYPE)
    Q_block  = tl.load(Q_blk,  boundary_check=(0, 1), padding_option="zero") * s
    dO_block = tl.load(dO_blk, boundary_check=(0, 1), padding_option="zero")
    dQ_block = tl.zeros((BLOCK_Q, HEAD_DIM), dtype=tl.float32)

    # ---------- loop over KV tiles ----------
    num_steps = tl.cdiv(SEQ_LEN, BLOCK_KV)
    for step in range(num_steps):
        K_T_block = tl.load(K_T_blk, boundary_check=(0, 1), padding_option="zero")
        V_T_block = tl.load(V_T_blk, boundary_check=(0, 1), padding_option="zero")
        
        start_kv = step * BLOCK_KV
        kv_idx   = start_kv + offs_kv
        kv_valid = kv_idx < SEQ_LEN
        S = tl.dot(Q_block, K_T_block)                     # [BLOCK_Q, BLOCK_KV]
        S = tl.where(kv_valid[None, :], S, -float("inf"))
        P = tl.exp(S - m)                                  # [BLOCK_Q, BLOCK_KV]

        # dP = dO @ Vᵀ  (match dtypes for dot)
        dP = tl.dot(dO_block.to(DTYPE), V_T_block.to(DTYPE)).to(tl.float32)
        dS = (P * (dP - Di[:, None])).to(DTYPE)
        dQ_block = tl.dot(dS, tl.trans(K_T_block.to(DTYPE)), dQ_block)

        K_T_blk = tl.advance(K_T_blk, (0, BLOCK_KV))
        V_T_blk = tl.advance(V_T_blk, (0, BLOCK_KV))
    
    dQ_block *= s
    tl.store(dQ_blk, dQ_block.to(dQ.type.element_ty), boundary_check=(0, 1))



class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.size()
        assert HEAD_DIM == V.size(-1) == K.size(-1)
        assert Q.is_contiguous()
        assert Q.stride() == K.stride() == V.stride()
        comp_torch = _sdpa_comp_dtype(Q)
        comp_triton = _triton_compute_dtype(comp_torch)
        
        softmax_scale = 1 / (HEAD_DIM**0.5)
        O = torch.empty_like(Q)

        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_Q"]),
            BATCH_SIZE * NUM_HEADS,
        )
        # M is the logsumexp for the backward pass, one for each query
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )
        _attn_fwd[grid](
            Q=Q, K=K, V=V, softmax_scale=softmax_scale, M=M, O=O,
            stride_batch=Q.stride(0), stride_head=Q.stride(1), stride_seq=Q.stride(2), stride_dim=Q.stride(3),
            BATCH_SIZE=Q.shape[0], NUM_HEADS=Q.shape[1], SEQ_LEN=Q.shape[2], HEAD_DIM=HEAD_DIM,
            DTYPE=comp_triton,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM
        ctx.comp_triton = comp_triton
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = ctx.saved_tensors
        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, _ = Q.size()
        BLOCK_MACRO = 128

        D = torch.empty_like(M)  # Shape: (BATCH_SIZE, NUM_HEADS, SEQ_LEN)
        # Compute all the elements Di
        preprocess_grid = (triton.cdiv(SEQ_LEN, BLOCK_MACRO), BATCH_SIZE * NUM_HEADS)
        _attn_bwd_preprocess[preprocess_grid](
            O=O, dO=dO, D=D, SEQ_LEN=SEQ_LEN,
            HEAD_DIM=ctx.HEAD_DIM,
        )
        grid = (triton.cdiv(SEQ_LEN, BLOCK_MACRO), BATCH_SIZE * NUM_HEADS)
        # Fix KV and iterate through all the Q blocks
        _attn_bwd_dk_dv[grid](
            Q=Q, K=K, V=V, softmax_scale=ctx.softmax_scale,
            dO=dO, dQ=dQ, dK=dK, dV=dV, M=M, D=D,
            stride_batch=Q.stride(0), stride_head=Q.stride(1), stride_seq=Q.stride(2), stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS, SEQ_LEN=SEQ_LEN, HEAD_DIM=ctx.HEAD_DIM, DTYPE=ctx.comp_triton,
        )

        # Fix Q and iterate through all the KV block
        _attn_bwd_dq[grid](
            Q=Q, K=K, V=V, softmax_scale=ctx.softmax_scale,
            dO=dO, dQ=dQ, dK=dK, dV=dV, M=M, D=D,
            stride_batch=Q.stride(0), stride_head=Q.stride(1), stride_seq=Q.stride(2), stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS, SEQ_LEN=SEQ_LEN, HEAD_DIM=ctx.HEAD_DIM, DTYPE=ctx.comp_triton,
        )
        return dQ, dK, dV, None, None
    
def sdpa_yt_fa(Q: Tensor, K: Tensor, V: Tensor):
    """ViT-S-only autograd op (single-pass forward + exact backward)."""
    return TritonAttention.apply(Q, K, V)


class SDPA_TRITON_FA(nn.Module):
    """
    Thin nn.Module wrapper around TritonAttention.apply(Q, K, V).

    Expects Q, K, V with shape [B, H, S, D] on the same device/dtype.
    Returns O with shape [B, H, S, D].
    """
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        if Q.dim() != 4:
            raise ValueError(f"Expected 4D tensors [B,H,S,D], got {Q.shape}, {K.shape}, {V.shape}")
        if K.shape != Q.shape or V.shape != Q.shape:
            raise ValueError(f"Q, K, V must have identical shapes; got {Q.shape}, {K.shape}, {V.shape}")
        if Q.device != K.device or Q.device != V.device:
            raise ValueError("Q, K, V must be on the same device")
        if Q.dtype != K.dtype or Q.dtype != V.dtype:
            raise ValueError("Q, K, V must have the same dtype")

        # Keep last dimension contiguous for clean strides (HEAD_DIM contiguous):
        if Q.stride(-1) != 1: Q = Q.contiguous()
        if K.stride(-1) != 1: K = K.contiguous()
        if V.stride(-1) != 1: V = V.contiguous()

        # Autograd is handled by your TritonAttention Function
        return TritonAttention.apply(Q, K, V)