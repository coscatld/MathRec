/** Bounded, in-process, FIFO executor. Node 24 native erasable TypeScript. */
export type Outcome<T> =
  | { ok: true; value: T }
  | { ok: false; code: 'INVALID_OPTIONS' | 'OVERLOADED' | 'TIMEOUT' | 'CANCELLED' | 'TASK_FAILED' };
export type Work<T> = (signal: AbortSignal) => T | PromiseLike<T>;
/** Trusted scheduler: never throws; callback runs later, not inside schedule(). */
export type Schedule = (callback: () => void, delayMs: number) => () => void;
export interface Clock { now: () => number; schedule: Schedule }
export const realClock: Clock = {
  now: () => performance.now(),
  schedule: (callback, delayMs) => {
    const timer = setTimeout(callback, delayMs);
    return () => clearTimeout(timer);
  },
};
interface Entry<T> {
  work: Work<T>;
  owned: AbortController;
  phase: 'queued' | 'running' | 'done';
  publicSettled: boolean;
  deadline: number;
  finish: (result: Outcome<T>) => boolean;
}
export class BoundedExecutor<T> {
  readonly #concurrency: number;
  readonly #maxQueue: number;
  readonly #clock: Clock;
  #queue: Entry<T>[] = [];
  #active = 0;
  #started = 0;
  #completed = 0;
  #overloaded = 0;
  #peakActive = 0;
  #peakQueued = 0;
  constructor(options: { concurrency: number; maxQueue: number }, clock: Clock = realClock) {
    const { concurrency, maxQueue } = options;
    if (!Number.isSafeInteger(concurrency) || concurrency < 1 || concurrency > 1024)
      throw new RangeError('concurrency must be an integer in [1, 1024]');
    if (!Number.isSafeInteger(maxQueue) || maxQueue < 0 || maxQueue > 100000)
      throw new RangeError('maxQueue must be an integer in [0, 100000]');
    this.#concurrency = concurrency;
    this.#maxQueue = maxQueue;
    this.#clock = clock;
  }
  snapshot() {
    return Object.freeze({ active: this.#active, queued: this.#queue.length,
      started: this.#started, completed: this.#completed, overloaded: this.#overloaded,
      peakActive: this.#peakActive, peakQueued: this.#peakQueued });
  }
  submit(work: Work<T>, options: { timeoutMs: number; signal?: AbortSignal }): Promise<Outcome<T>> {
    const { timeoutMs, signal: caller } = options;
    if (!Number.isInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 60000)
      return Promise.resolve({ ok: false, code: 'INVALID_OPTIONS' });
    if (caller?.aborted) return Promise.resolve({ ok: false, code: 'CANCELLED' });
    if (this.#active >= this.#concurrency && this.#queue.length >= this.#maxQueue) {
      this.#overloaded++;
      return Promise.resolve({ ok: false, code: 'OVERLOADED' });
    }
    return new Promise<Outcome<T>>((resolve) => {
      const entry: Entry<T> = { work, owned: new AbortController(), phase: 'queued',
        publicSettled: false, deadline: this.#clock.now() + timeoutMs, finish: () => false };
      let cancelTimer = () => {};
      const onAbort = () => stop('CANCELLED');
      entry.finish = (result) => {
        if (entry.publicSettled) return false;
        entry.publicSettled = true;
        cancelTimer();
        caller?.removeEventListener('abort', onAbort);
        resolve(result);
        return true;
      };
      const stop = (code: 'TIMEOUT' | 'CANCELLED') => {
        if (entry.publicSettled) return;
        if (entry.phase === 'queued') {
          const index = this.#queue.indexOf(entry);
          if (index !== -1) this.#queue.splice(index, 1);
          entry.phase = 'done'; // Never invoked: no underlying work to release.
          entry.finish({ ok: false, code });
        } else if (entry.phase === 'running') {
          // Public result first. Abort may cause the backend to reject immediately.
          entry.finish({ ok: false, code });
          entry.owned.abort();
          // DO NOT decrement active here. Only #complete owns that operation.
        }
      };
      caller?.addEventListener('abort', onAbort, { once: true });
      const checkDeadline = () => {
        const remaining = entry.deadline - this.#clock.now();
        if (remaining <= 0) stop('TIMEOUT');
        else cancelTimer = this.#clock.schedule(checkDeadline, Math.ceil(remaining));
      };
      cancelTimer = this.#clock.schedule(checkDeadline, timeoutMs);
      if (this.#active < this.#concurrency) this.#start(entry);
      else {
        this.#queue.push(entry);
        this.#peakQueued = Math.max(this.#peakQueued, this.#queue.length);
      }
    });
  }
  #start(entry: Entry<T>) {
    // A delayed timer must not let expired queued work reach the backend.
    if (this.#clock.now() >= entry.deadline) {
      entry.phase = 'done';
      entry.finish({ ok: false, code: 'TIMEOUT' });
      return;
    }
    entry.phase = 'running';
    this.#active++;
    this.#started++;
    this.#peakActive = Math.max(this.#peakActive, this.#active);
    let work: Promise<T>;
    try { work = Promise.resolve(entry.work(entry.owned.signal)); }
    catch (error) { work = Promise.reject(error); }
    // Both handlers remain attached after timeout; late rejection is observed.
    work.then(
      (value) => this.#complete(entry, { ok: true, value }),
      () => this.#complete(entry, { ok: false, code: 'TASK_FAILED' }),
    );
  }
  #complete(entry: Entry<T>, result: Outcome<T>) {
    entry.phase = 'done';
    this.#active--;
    this.#completed++;
    // Completion observed at/after the deadline is TIMEOUT, even if its timer lagged.
    entry.finish(this.#clock.now() >= entry.deadline
      ? { ok: false, code: 'TIMEOUT' } : result);
    // A previous timeout/cancellation result cannot be overwritten.
    while (this.#active < this.#concurrency && this.#queue.length > 0)
      this.#start(this.#queue.shift()!);
  }
}
