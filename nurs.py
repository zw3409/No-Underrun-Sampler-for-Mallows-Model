from __future__ import annotations
import math
from typing import Optional, Tuple, Optional, Union
import numba as nb
import numpy as np
from distance import dist, invert, DIST_NAME_TO_ID


@nb.njit(cache=True, fastmath=True, inline="always")
def logsumexp2(a: float, b: float) -> float:
    return a + math.log1p(math.exp(b - a)) if a > b else b + math.log1p(math.exp(a - b))


def sample_uniform(n: int) -> np.ndarray:
    return np.random.permutation(n).astype(np.int64)

def sample_ql(n: int, l: int) -> np.ndarray:
    rho = np.arange(n, dtype=np.int64)
    start = np.random.randint(0, n - l + 1)
    idx = np.arange(start, start + l, dtype=np.int64)
    rho[idx[:-1]] = idx[1:]
    rho[idx[-1]] = idx[0]
    return rho


def sample_qw(n: int, w: int) -> np.ndarray:
    rho = np.arange(n, dtype=np.int64)
    u = np.random.randint(0, w)
    idx = np.arange(n, dtype=np.int64)
    rolled = np.roll(idx, -u)
    for s in range(0, n, w):
        e = min(s + w, n)
        block = rolled[s:e]
        rho[block] = block[np.random.permutation(e - s)]
    return rho


@nb.njit(cache=True, fastmath=True, inline="always")
def apply_perm(out: np.ndarray, src: np.ndarray, perm: np.ndarray) -> None:
    for ii in range(src.size):
        out[ii] = src[perm[ii]]


@nb.njit(cache=True, fastmath=True)
def sub_block_stops_log(log_w: np.ndarray, log_eps: float) -> bool:
    block = log_w.size
    while block >= 1:
        for start in range(0, log_w.size, block):
            end = start + block
            if end > log_w.size:
                break
            log_sum = log_w[start]
            for k2 in range(start + 1, end):
                log_sum = logsumexp2(log_sum, log_w[k2])
            if max(log_w[start], log_w[end - 1]) <= log_eps + log_sum:
                return True
        block >>= 1
    return False


@nb.njit(cache=True, fastmath=True)
def nurs_kernel(
    n: int,
    start_perm: np.ndarray,
    beta: float,
    eps: float,
    max_doublings: int,
    rho: np.ndarray,
    dist_id: int,
) -> np.ndarray:
    log_eps = math.log(eps)
    orbit_left = start_perm.copy()
    orbit_right = start_perm.copy()
    chosen_state = start_perm.copy()
    total_log_weight = -beta * dist(dist_id, start_perm)
    step_fwd = rho
    step_bwd = invert(step_fwd)
    perm_buffer = np.empty(n, dtype=np.int64)
    doubling_bits = np.random.randint(0, 2, size=max_doublings)
    for j in range(max_doublings):
        extension_len = 1 << j
        grow_forward = doubling_bits[j] == 1
        anchor = orbit_right if grow_forward else orbit_left
        step_perm = step_fwd if grow_forward else step_bwd
        ext_logw = np.empty(extension_len, np.float64)
        perm_curr = anchor.copy()
        for t in range(extension_len):
            apply_perm(perm_buffer, perm_curr, step_perm)
            perm_curr, perm_buffer = perm_buffer, perm_curr
            ext_logw[t] = -beta * dist(dist_id, perm_curr)
        if sub_block_stops_log(ext_logw, log_eps):
            break
        ext_last_state = perm_curr.copy()
        perm_curr = anchor.copy()
        for t in range(extension_len):
            apply_perm(perm_buffer, perm_curr, step_perm)
            perm_curr, perm_buffer = perm_buffer, perm_curr
            log_w = ext_logw[t]
            new_total = logsumexp2(total_log_weight, log_w)
            if np.random.random() < math.exp(min(0.0, log_w - new_total)):
                chosen_state = perm_curr.copy()
            total_log_weight = new_total
        if grow_forward:
            orbit_right = ext_last_state
        else:
            orbit_left = ext_last_state
        if max(-beta * dist(dist_id, orbit_left), -beta * dist(dist_id, orbit_right)) <= log_eps + total_log_weight:
            break
    return chosen_state


def sample(
    n: int,
    sigma0: Tuple[int, ...],
    beta: float,
    eps: float,
    max_doublings: int,
    dist_id: int | str,
    restricted_i: int | None = None,
    w: Optional[int] = None,
) -> Tuple[int, ...]:
    sigma0_arr = np.fromiter(sigma0, dtype=np.int64, count=n)
    if restricted_i is None:
        rho_arr = sample_uniform(n)
    if isinstance(dist_id, str):
        dist_id = DIST_NAME_TO_ID[dist_id]
    next_state = nurs_kernel(
        n=n,
        start_perm=sigma0_arr,
        beta=beta,
        eps=eps,
        max_doublings=max_doublings,
        rho=rho_arr,
        dist_id=dist_id,
    )
    return tuple(int(x) for x in next_state)

def sample(
    n: int,
    sigma0: Tuple[int, ...],
    beta: float,
    eps: float,
    max_doublings: int,
    dist_id: Union[int, str],
    w: Optional[int] = None,
) -> Tuple[int, ...]:
    sigma0_arr = np.fromiter(sigma0, dtype=np.int64, count=n)
    if w is None:
        rho_arr = sample_uniform(n)
    else:
        rho_arr = sample_qw(n, w)
    if isinstance(dist_id, str):
        dist_id = DIST_NAME_TO_ID[dist_id]
    next_state = nurs_kernel(
        n=n,
        start_perm=sigma0_arr,
        beta=beta,
        eps=eps,
        max_doublings=max_doublings,
        rho=rho_arr,
        dist_id=dist_id,
    )
    return tuple(int(x) for x in next_state)



__all__ = ["sample", "sample_uniform", "sample_qw","sample_ql", "nurs_kernel"]

