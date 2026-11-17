import { parseRequest, parseReply, type Request, type Reply } from './contract.ts';
export interface CancellableBackend {
  generate(request: Request, signal: AbortSignal): Promise<unknown>;
}
export type Outcome =
  | { ok: true; reply: Reply }
  | { ok: false; code: 'INVALID_INPUT' | 'INVALID_OPTIONS' | 'BACKEND_FAILED'
      | 'INVALID_OUTPUT' | 'TIMEOUT' | 'CANCELLED' };
// Trusted scheduler: callback runs later, never inside schedule(); returns cleanup.
export type Schedule = (callback: () => void, delayMs: number) => () => void;
export const realSchedule: Schedule = (callback, delayMs) => {
  const timer = setTimeout(callback, delayMs);
  return () => clearTimeout(timer);
};
export function answerWithDeadline(
  input: unknown,
  backend: CancellableBackend,
  options: { timeoutMs: number; signal?: AbortSignal },
  schedule: Schedule = realSchedule,
): Promise<Outcome> {
  const request = parseRequest(input);
  if (request === null) return Promise.resolve({ ok: false, code: 'INVALID_INPUT' });
  const { timeoutMs, signal: caller } = options;
  if (!Number.isInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 60000)
    return Promise.resolve({ ok: false, code: 'INVALID_OPTIONS' });
  if (caller?.aborted) return Promise.resolve({ ok: false, code: 'CANCELLED' });
  const owned = new AbortController();
  return new Promise<Outcome>((resolve) => {
    let settled = false;
    let cancelTimer = () => {};
    const onCallerAbort = () => stop('CANCELLED');
    function finish(result: Outcome): boolean {
      if (settled) return false;
      settled = true;
      cancelTimer();
      caller?.removeEventListener('abort', onCallerAbort);
      resolve(result);
      return true;
    }
    function stop(code: 'TIMEOUT' | 'CANCELLED') {
      // Choose public outcome before cooperative abort can reject the backend.
      if (finish({ ok: false, code })) owned.abort();
    }
    caller?.addEventListener('abort', onCallerAbort, { once: true });
    cancelTimer = schedule(() => stop('TIMEOUT'), timeoutMs);
    // Both handlers remain attached even after the public promise has settled.
    try {
      Promise.resolve(backend.generate(request, owned.signal)).then(
        (raw) => {
          if (settled) return; // discard late value, no further public result
          const reply = parseReply(raw, request);
          finish(reply === null ? { ok: false, code: 'INVALID_OUTPUT' } : { ok: true, reply });
        },
        () => { finish({ ok: false, code: 'BACKEND_FAILED' }); },
      );
    } catch {
      finish({ ok: false, code: 'BACKEND_FAILED' });
    }
  });
}
