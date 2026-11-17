import { performance } from 'node:perf_hooks';

const CALLER = new Set(['ok', 'error', 'timeout', 'cancelled']);
const BACKEND = new Set(['ok', 'error', 'cancelled']);

/** Exact nearest-rank quantile, not an interpolated or streaming estimate. */
export function quantile(values, p) {
  if (!(p > 0 && p <= 1)) throw new RangeError('p must be in (0, 1]');
  if (values.some(x => !Number.isFinite(x) || x < 0)) throw new RangeError('invalid duration');
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.ceil(p * sorted.length) - 1];
}
export function distribution(values) {
  return { count: values.length, p50: quantile(values, .5),
    p95: quantile(values, .95), p99: quantile(values, .99) };
}

/** One process, one monotonic clock, finite teaching batch; times in ms. */
export class Recorder {
  #clock; #last = -Infinity; #rows = new Map();
  constructor(clock = () => performance.now()) { this.#clock = clock; }
  #now() {
    const t = this.#clock();
    if (!Number.isFinite(t) || t < 0 || t < this.#last) throw new Error('non-monotonic clock');
    this.#last = t;
    return t;
  }
  #get(id) {
    const r = this.#rows.get(id);
    if (!r) throw new Error('unknown request');
    return r;
  }
  accept(id) {
    if (typeof id !== 'string' || !id || this.#rows.has(id)) throw new Error('invalid or duplicate id');
    this.#rows.set(id, { id, acceptedAt: this.#now(), startedAt: null,
      backendAt: null, backendOutcome: null, callerAt: null, callerOutcome: null });
  }
  start(id) {
    const r = this.#get(id);
    if (r.startedAt !== null || r.callerAt !== null) throw new Error('request is not queued');
    r.startedAt = this.#now();
  }
  finishBackend(id, outcome) {
    const r = this.#get(id);
    if (!BACKEND.has(outcome) || r.startedAt === null || r.backendAt !== null)
      throw new Error('invalid backend transition');
    r.backendAt = this.#now();
    r.backendOutcome = outcome;
  }
  finishCaller(id, outcome) {
    const r = this.#get(id);
    if (!CALLER.has(outcome) || r.callerAt !== null) throw new Error('invalid caller transition');
    if ((outcome === 'ok' || outcome === 'error') &&
        (r.backendAt === null || outcome !== r.backendOutcome))
      throw new Error('caller result requires matching completed backend');
    r.callerAt = this.#now();
    r.callerOutcome = outcome;
  }
  snapshot() { return [...this.#rows.values()].map(r => ({ ...r })); }
  report() {
    const r = this.snapshot();
    // Do not put absent measurements into the distribution as zero.
    const started = r.filter(x => x.startedAt !== null);
    const abandoned = r.filter(x => x.startedAt === null && x.callerAt !== null);
    const backend = r.filter(x => x.backendAt !== null);
    const caller = r.filter(x => x.callerAt !== null);
    return {
      accepted: r.length,
      gauges: {
        queued: r.filter(x => x.startedAt === null && x.callerAt === null).length,
        running: r.filter(x => x.startedAt !== null && x.backendAt === null).length,
        callerPending: r.filter(x => x.callerAt === null).length,
      },
      callerOutcomes: Object.fromEntries([...CALLER].map(k => [k, caller.filter(x => x.callerOutcome === k).length])),
      backendOutcomes: Object.fromEntries([...BACKEND].map(k => [k, backend.filter(x => x.backendOutcome === k).length])),
      milliseconds: {
        startedQueueWait: distribution(started.map(x => x.startedAt - x.acceptedAt)),
        abandonedQueueWait: distribution(abandoned.map(x => x.callerAt - x.acceptedAt)),
        execution: distribution(backend.map(x => x.backendAt - x.startedAt)),
        callerWait: distribution(caller.map(x => x.callerAt - x.acceptedAt)),
      },
    };
  }
}
