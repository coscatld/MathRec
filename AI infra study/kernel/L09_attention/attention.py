"""Single-head attention: transparent CPU reference, Python 3.10+, stdlib only.

Keep mask: True means allowed. Fully masked rows produce exact zeros.
Teaching limits prevent nonfinite intermediate arithmetic, not resource exhaustion.
"""
from dataclasses import dataclass
import math

MAX_DIM = 256
MAX_ABS = 1e6


@dataclass
class Trace:
    scores: list[list[float]]          # Q K^T / sqrt(D), BEFORE mask
    masked_scores: list[list[float]]   # forbidden scores replaced with -inf
    probabilities: list[list[float]]
    output: list[list[float]]


def _matrix(x, name):
    if not isinstance(x, (list, tuple)) or not 1 <= len(x) <= MAX_DIM:
        raise ValueError(f'{name}: require 1..{MAX_DIM} rows')
    width = None
    for row in x:
        if not isinstance(row, (list, tuple)) or not 1 <= len(row) <= MAX_DIM:
            raise ValueError(f'{name}: require nonempty bounded rows')
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError(f'{name}: ragged matrix')
        for value in row:
            if type(value) not in (int, float):
                raise ValueError(f'{name}: only int/float scalars, not bool')
            # Compare before conversion so arbitrarily large Python ints reject safely.
            if not -MAX_ABS <= value <= MAX_ABS or not math.isfinite(value):
                raise ValueError(f'{name}: require finite values within +/-{MAX_ABS}')
    return width


def _validate(q, k, v, keep):
    d = _matrix(q, 'Q')
    if _matrix(k, 'K') != d:
        raise ValueError('Q and K widths differ')
    dv = _matrix(v, 'V')
    if len(k) != len(v):
        raise ValueError('K and V row counts differ')
    if keep is not None:
        if not isinstance(keep, (list, tuple)) or len(keep) != len(q):
            raise ValueError('keep must have shape [L,S]')
        for row in keep:
            if (not isinstance(row, (list, tuple)) or len(row) != len(k)
                    or any(type(b) is not bool for b in row)):
                raise ValueError('keep must have shape [L,S] with strict bool entries')
    return d, dv


def causal_keep(query_positions, key_positions):
    """Absolute positions only: key_pos <= query_pos. No implicit L/S alignment.

    Decode example: query_positions=[5], key_positions=list(range(6)).
    Position order may be arbitrary; combining padding uses Boolean AND outside.
    """
    for positions in (query_positions, key_positions):
        if not isinstance(positions, (list, tuple)) or not 1 <= len(positions) <= MAX_DIM:
            raise ValueError('positions must be nonempty bounded sequences')
        if any(type(p) is not int or not 0 <= p <= 10**9 for p in positions):
            raise ValueError('positions must be nonnegative bounded integers')
    return [[kp <= qp for kp in key_positions] for qp in query_positions]


def _score(qrow, krow, scale):
    return math.fsum(a * b for a, b in zip(qrow, krow)) * scale


def attention_materialized(q, k, v, keep=None):
    """Return all intermediate matrices, for inspection rather than memory savings."""
    d, dv = _validate(q, k, v, keep)
    scale = 1.0 / math.sqrt(d)
    scores, masked, probs, outputs = [], [], [], []
    for i, qrow in enumerate(q):
        row = [_score(qrow, krow, scale) for krow in k]
        allowed = [True] * len(k) if keep is None else keep[i]
        masked_row = [s if ok else -math.inf for s, ok in zip(row, allowed)]
        if not any(allowed):
            p = [0.0] * len(k)  # Never evaluate -inf - (-inf).
        else:
            m = max(masked_row)
            weights = [math.exp(s - m) if ok else 0.0
                       for s, ok in zip(row, allowed)]
            denominator = math.fsum(weights)
            p = [w / denominator for w in weights]
        output = [math.fsum(p[j] * v[j][c] for j in range(len(k)))
                  for c in range(dv)]
        scores.append(row)
        masked.append(masked_row)
        probs.append(p)
        outputs.append(output)
    return Trace(scores, masked, probs, outputs)


def attention_streaming(q, k, v, keep=None, block_size=4):
    """Return O without an L*S score/probability allocation.

    Per row state after processed allowed keys:
      m = maximum score; l = sum exp(score-m)
      acc[c] = sum exp(score-m)*V[key,c]; O[c] = acc[c]/l.
    Rescale BOTH l and acc whenever the running maximum changes.
    Workspace excludes input, mask and output: O(block_size + Dv).
    """
    d, dv = _validate(q, k, v, keep)
    if type(block_size) is not int or not 1 <= block_size <= MAX_DIM:
        raise ValueError(f'block_size must be an integer in 1..{MAX_DIM}')
    scale = 1.0 / math.sqrt(d)
    outputs = []
    for i, qrow in enumerate(q):
        m, l = -math.inf, 0.0
        acc = [0.0] * dv
        for start in range(0, len(k), block_size):
            indices = [j for j in range(start, min(start + block_size, len(k)))
                       if keep is None or keep[i][j]]
            if not indices:
                continue  # Includes a fully masked FIRST block.
            scores = [_score(qrow, k[j], scale) for j in indices]
            new_m = max(m, max(scores))
            alpha = 0.0 if l == 0.0 else math.exp(m - new_m)
            weights = [math.exp(s - new_m) for s in scores]
            l = alpha * l + math.fsum(weights)
            acc = [alpha * acc[c] + math.fsum(w * v[j][c]
                   for w, j in zip(weights, indices)) for c in range(dv)]
            m = new_m
        outputs.append([a / l for a in acc] if l else [0.0] * dv)
    return outputs
