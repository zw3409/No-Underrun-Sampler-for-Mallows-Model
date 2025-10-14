from __future__ import annotations
import math
from typing import Optional, Tuple

import numba as nb
import numpy as np

from distance import dist, invert, DIST_NAME_TO_ID


@nb.njit(cache=True, fastmath=True, inline="always")
def logsumexp2(a: float, b: float) -> float:
    return a + math.log1p(math.exp(b - a)) if a > b else b + math.log1p(math.exp(a - b))


def sample_unif(n: int) -> np.ndarray:
    perm = np.arange(n, dtype=np.int64)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def odd_prime_powers(L: int) -> np.ndarray:
    x = L
    res = []
    p = 3
    while p * p <= x:
        if x % p == 0:
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            res.append(p ** e)
        p += 2
    if x > 1:
        res.append(x)
    return np.array(res, dtype=np.int64)


def unbounded_sum_reconstruct_max(
    total: int, allowed: np.ndarray, rng: np.random.Generator
) -> Tuple[list[int], int]:
    total = int(total)
    if total <= 0 or allowed.size == 0:
        return [], 0
    allowed = np.asarray(allowed, dtype=np.int64)
    allowed = allowed[allowed > 0]
    if allowed.size == 0:
        return [], 0
    order = allowed.copy()
    rng.shuffle(order)
    can = np.zeros(total + 1, dtype=np.bool_)
    prev = np.full(total + 1, -1, dtype=np.int64)
    take = np.full(total + 1, -1, dtype=np.int64)
    can[0] = True
    for s in order:
        s = int(s)
        for t in range(s, total + 1):
            if (not can[t]) and can[t - s]:
                can[t] = True
                prev[t] = t - s
                take[t] = s
    y = total
    while y >= 0 and not can[y]:
        y -= 1
    if y <= 0:
        return [], 0
    res = []
    t = y
    while t > 0:
        s = int(take[t])
        res.append(s)
        t = int(prev[t])
    return res, y


def sample_eta_fixed_order(
    n: int,
    L_target: int,
    i: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    k: int = 3,
) -> Tuple[np.ndarray, int]:
    rng = np.random.default_rng() if rng is None else rng
    if i is None:
        i = int(rng.integers(0, n - 1))
    pool = np.concatenate((np.arange(i, dtype=np.int64), np.arange(i + 2, n, dtype=np.int64)))
    rng.shuffle(pool)
    m = pool.size
    if m == 0:
        h = np.arange(n, dtype=np.int64)
        h[i] = i
        h[i + 1] = i + 1
        eta = h.copy()
        mask_i = eta == i
        mask_ip1 = eta == i + 1
        eta[mask_i] = i + 1
        eta[mask_ip1] = i
        return eta, 1
    min_odd = 3 if k <= 1 else k
    allowed_no1 = np.array([s for s in range(min_odd, m + 1, 2) if (L_target % s) == 0], dtype=np.int64)
    pp = odd_prime_powers(L_target)
    sizes_base: list[int] = []
    for q in pp:
        cand = allowed_no1[allowed_no1 % int(q) == 0]
        if cand.size == 0:
            raise ValueError(
                f"Impossible under fixed order {L_target}: need a cycle multiple of {q} but none ≤ m={m} (k={k})."
            )
        sizes_base.append(int(np.min(cand)))
    S_base = int(sum(sizes_base))
    if S_base > m:
        raise ValueError(f"Impossible: minimal coverage sum {S_base} exceeds pool size m={m} for order {L_target}.")
    rem = m - S_base
    fill_list, reached = unbounded_sum_reconstruct_max(rem, allowed_no1, rng)
    ones_needed = rem - reached
    sizes = np.array(sizes_base + fill_list + [1] * int(ones_needed), dtype=np.int64)
    if int(np.sum(sizes)) != m:
        raise RuntimeError("Internal error: sizes do not sum to pool size m.")
    h = np.arange(n, dtype=np.int64)
    p = 0
    order_idx = rng.permutation(sizes.size)
    for idx in order_idx:
        s = int(sizes[idx])
        block = pool[p : p + s]
        rng.shuffle(block)
        h[block] = np.roll(block, -1)
        p += s
    h[i] = i
    h[i + 1] = i + 1
    eta = h.copy()
    mask_i = eta == i
    mask_ip1 = eta == i + 1
    eta[mask_i] = i + 1
    eta[mask_ip1] = i
    L_out = 1
    for s in sizes:
        L_out = math.lcm(L_out, int(s))
    if L_out != L_target:
        raise RuntimeError(f"Construction error: lcm(sizes)={L_out} != target {L_target}")
    return eta, L_out


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
) -> Tuple[int, ...]:
    sigma0_arr = np.fromiter(sigma0, dtype=np.int64, count=n)
    if restricted_i is None:
        rho_arr = sample_unif(n)
    else:
        rho_arr, _ = sample_eta_fixed_order(n=n, L_target=restricted_i)
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


__all__ = ["sample", "sample_unif", "nurs_kernel", "sample_eta", "sample_eta_fixed_order"]
