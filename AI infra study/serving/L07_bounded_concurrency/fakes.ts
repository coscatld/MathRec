import type { Schedule } from './executor.ts';
export class ManualClock {
  #now = 0;
  #next = 0;
  #timers = new Map<number, { at: number; callback: () => void }>();
  schedule: Schedule = (callback, delayMs) => {
    const id = this.#next++;
    this.#timers.set(id, { at: this.#now + delayMs, callback });
    return () => { this.#timers.delete(id); };
  };
  now = () => this.#now;
  elapseWithoutTimers(ms: number) {
    if (!Number.isFinite(ms) || ms < 0) throw new RangeError('invalid time step');
    this.#now += ms;
  }
  get pending() { return this.#timers.size; }
  advance(ms: number) {
    if (!Number.isFinite(ms) || ms < 0) throw new RangeError('invalid time step');
    this.#now += ms;
    while (true) {
      const due = [...this.#timers].filter(([, t]) => t.at <= this.#now)
        .sort(([a, x], [b, y]) => x.at - y.at || a - b)[0];
      if (!due) break;
      const [id, timer] = due;
      this.#timers.delete(id);
      timer.callback();
    }
  }
}
export interface Call {
  id: string;
  signal: AbortSignal;
  aborted: boolean;
  finish: (value?: string) => void;
  fail: () => void;
}
export class ControlledBackend {
  calls: Call[] = [];
  active = 0;
  peak = 0;
  generate(id: string, signal: AbortSignal, cooperative = false): Promise<string> {
    this.active++;
    this.peak = Math.max(this.peak, this.active);
    return new Promise((resolve, reject) => {
      let ended = false;
      const cleanup = () => {
        if (ended) return false;
        ended = true;
        this.active--;
        signal.removeEventListener('abort', onAbort);
        return true;
      };
      const call: Call = { id, signal, aborted: signal.aborted,
        finish: (value = `reply:${id}`) => { if (cleanup()) resolve(value); },
        fail: () => { if (cleanup()) reject(new Error('mock backend failure')); } };
      const onAbort = () => {
        call.aborted = true;
        if (cooperative) call.fail();
      };
      this.calls.push(call);
      signal.addEventListener('abort', onAbort, { once: true });
      if (signal.aborted) onAbort();
    });
  }
}
export async function flush() {
  // Test-only: let the finite promise continuations in these mocks run.
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}
