from online_softmax import State, block_state, merge, stable_logsumexp, two_pass_softmax

if __name__ == '__main__':
    xs = [1000.0, 1001.0, 999.0, -1000.0, 1002.0]
    a, b = block_state(xs[:2]), block_state(xs[2:])
    print('blocks:', a, b)
    print('merged:', merge(a, b))
    print('empty identity:', merge(State(), a))
    print('logsumexp:', stable_logsumexp(xs))
    print('softmax:', two_pass_softmax(xs, block_size=2))
    print('large offset preserves normalization:', two_pass_softmax([1e30, 1e30]))
