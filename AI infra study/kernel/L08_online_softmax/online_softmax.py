"""Dependency-free finite-real, forward-only reference (Python binary64)."""
from dataclasses import dataclass
import math
from numbers import Real


def finite_values(values):
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError('inputs must be real numbers, not booleans')
        x = float(value)
        if not math.isfinite(x):
            raise ValueError('inputs must be finite')
        result.append(x)
    return result


@dataclass(frozen=True)
class State:
    """m=max(x), l=sum(exp(x-m)); (-inf, 0) is the empty identity."""
    m: float = -math.inf
    l: float = 0.0

    def __post_init__(self):
        if not ((self.m == -math.inf and self.l == 0.0) or
                (math.isfinite(self.m) and math.isfinite(self.l) and self.l >= 1.0)):
            raise ValueError('invalid normalizer state')

    @property
    def logsumexp(self):
        return -math.inf if self.l == 0 else self.m + math.log(self.l)


def block_state(values):
    xs = finite_values(values)
    if not xs:
        return State()
    m = max(xs)
    return State(m, math.fsum(math.exp(x - m) for x in xs))


def merge(a, b):
    # Branch before subtracting maxima: (-inf)-(-inf) would be NaN.
    if a.l == 0:
        return b
    if b.l == 0:
        return a
    m = max(a.m, b.m)
    return State(m, a.l * math.exp(a.m - m) + b.l * math.exp(b.m - m))


def online_normalizer(blocks):
    """Consumes blocks once; retains O(1) state (plus current block)."""
    state = State()
    for block in blocks:
        state = merge(state, block_state(block))
    return state


def stable_logsumexp(values):
    """Empty input returns -inf, the logarithm of an empty exponential sum."""
    return block_state(values).logsumexp


def two_pass_softmax(values, block_size=128):
    """Materializes a reusable input, normalizes blocks, then emits probabilities.

    This is not a one-pass streaming output algorithm: earlier probabilities
    depend on later inputs. Empty softmax is undefined and rejected.
    """
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError('block_size must be a positive integer')
    xs = finite_values(values)
    if not xs:
        raise ValueError('softmax needs at least one value')
    state = online_normalizer(xs[i:i + block_size] for i in range(0, len(xs), block_size))
    # Do not reconstruct exp(x-lse): m+log(l) can lose log(l) for large offsets.
    return [math.exp(x - state.m) / state.l for x in xs]
