import type { Schedule, CancellableBackend } from './timeout.ts';
export function manualTimer() {
  let callback: (() => void) | undefined;
  let active = false;
  const schedule: Schedule = (cb) => {
    callback = cb; active = true;
    return () => { active = false; };
  };
  return { schedule, fire: () => { if (active) callback?.(); }, active: () => active };
}
export function controlledBackend(cooperative: boolean) {
  let resolve!: (value: unknown) => void;
  let reject!: (reason: unknown) => void;
  let calls = 0;
  let stopped = false;
  let seen: AbortSignal | undefined;
  let clean = () => {};
  const backend: CancellableBackend = {
    generate(_request, signal) {
      calls++; seen = signal;
      return new Promise((res, rej) => {
        resolve = res; reject = rej;
        if (cooperative) {
          const onAbort = () => { stopped = true; rej(new Error('mock aborted')); };
          if (signal.aborted) onAbort();
          else signal.addEventListener('abort', onAbort, { once: true });
          clean = () => signal.removeEventListener('abort', onAbort);
        }
      });
    },
  };
  return { backend,
    complete(value: unknown) { stopped = true; clean(); resolve(value); },
    fail() { stopped = true; clean(); reject(new Error('private provider detail')); },
    calls: () => calls, stopped: () => stopped, signal: () => seen };
}
