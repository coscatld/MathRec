"""Run: python example.py. Shows unequal L/S absolute-position causal decode."""
from pprint import pprint
from attention import attention_materialized, attention_streaming, causal_keep


def main():
    q = [[1., 0.], [0., 1.]]
    k = [[1., 0.], [0., 1.], [1., 1.], [-1., 1.], [2., 0.]]
    v = [[10., 0.], [0., 20.], [30., 30.], [40., 10.], [99., 99.]]
    keep = causal_keep([2, 3], [0, 1, 2, 3, 4])
    trace = attention_materialized(q, k, v, keep)
    for name, value in [('keep (True=allowed)', keep), ('scaled scores', trace.scores),
                        ('masked scores', trace.masked_scores),
                        ('probabilities', trace.probabilities), ('O', trace.output),
                        ('streamed O, block=2', attention_streaming(q, k, v, keep, 2))]:
        print('\n' + name)
        pprint(value)


if __name__ == '__main__':
    main()
