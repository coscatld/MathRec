import test from 'node:test';
import assert from 'node:assert/strict';
import { getEventListeners } from 'node:events';
import { answerWithDeadline } from './timeout.ts';
import { manualTimer, controlledBackend } from './fakes.ts';
const input = { requestId: 'r1', prompt: 'hello', maxOutputChars: 20 };
const reply = { requestId: 'r1', text: 'mock:hello', finishReason: 'stop' };
const tick = () => new Promise<void>((r) => setImmediate(r));
test('success and cleanup', async () => {
  const t = manualTimer(), b = controlledBackend(true), c = new AbortController();
  const p = answerWithDeadline(input, b.backend, { timeoutMs: 10, signal: c.signal }, t.schedule);
  assert.equal(getEventListeners(c.signal, 'abort').length, 1);
  b.complete(reply); assert.deepEqual(await p, { ok: true, reply });
  assert.equal(t.active(), false); assert.equal(getEventListeners(c.signal, 'abort').length, 0);
  assert.equal(b.signal()?.aborted, false);
});
test('invalid input/options do not start work', async () => {
  const b = controlledBackend(false);
  assert.equal((await answerWithDeadline({}, b.backend, { timeoutMs: 10 })).ok, false);
  for (const timeoutMs of [0, -1, NaN, Infinity, 1.5, 60001])
    assert.deepEqual(await answerWithDeadline(input, b.backend, { timeoutMs }), { ok: false, code: 'INVALID_OPTIONS' });
  assert.equal(b.calls(), 0);
});
test('pre-aborted caller does not start work', async () => {
  const b = controlledBackend(false), c = new AbortController(); c.abort();
  assert.deepEqual(await answerWithDeadline(input, b.backend, { timeoutMs: 10, signal: c.signal }), { ok: false, code: 'CANCELLED' });
  assert.equal(b.calls(), 0);
});
test('caller cancellation requests cooperative stop', async () => {
  const t = manualTimer(), b = controlledBackend(true), c = new AbortController();
  const p = answerWithDeadline(input, b.backend, { timeoutMs: 10, signal: c.signal }, t.schedule);
  c.abort(); assert.deepEqual(await p, { ok: false, code: 'CANCELLED' });
  assert.equal(b.stopped(), true); assert.equal(t.active(), false);
  assert.equal(getEventListeners(c.signal, 'abort').length, 0);
});
test('timeout wins over rejection caused by cooperative abort', async () => {
  const t = manualTimer(), b = controlledBackend(true), c = new AbortController();
  const p = answerWithDeadline(input, b.backend, { timeoutMs: 10, signal: c.signal }, t.schedule);
  t.fire(); assert.deepEqual(await p, { ok: false, code: 'TIMEOUT' });
  assert.equal(b.stopped(), true); assert.equal(c.signal.aborted, false);
  assert.equal(getEventListeners(c.signal, 'abort').length, 0);
});
test('uncooperative work continues after timeout; late success discarded', async () => {
  const t = manualTimer(), b = controlledBackend(false);
  const p = answerWithDeadline(input, b.backend, { timeoutMs: 10 }, t.schedule);
  t.fire(); assert.deepEqual(await p, { ok: false, code: 'TIMEOUT' });
  assert.equal(b.stopped(), false); assert.equal(b.signal()?.aborted, true);
  b.complete(reply); await tick(); assert.deepEqual(await p, { ok: false, code: 'TIMEOUT' });
});
test('late rejection is observed, no unhandledRejection', async () => {
  const t = manualTimer(), b = controlledBackend(false);
  const p = answerWithDeadline(input, b.backend, { timeoutMs: 10 }, t.schedule);
  t.fire(); await p; b.fail(); await tick(); // Node test runner also fails unhandled rejection.
  assert.deepEqual(await p, { ok: false, code: 'TIMEOUT' });
});
test('synchronous throw normalized', async () => {
  const t = manualTimer();
  assert.deepEqual(await answerWithDeadline(input, { generate() { throw new Error('secret'); } }, { timeoutMs: 10 }, t.schedule), { ok: false, code: 'BACKEND_FAILED' });
  assert.equal(t.active(), false);
});
test('asynchronous rejection normalized', async () => {
  const t = manualTimer(), b = controlledBackend(false);
  const p = answerWithDeadline(input, b.backend, { timeoutMs: 10 }, t.schedule);
  b.fail(); assert.deepEqual(await p, { ok: false, code: 'BACKEND_FAILED' }); assert.equal(t.active(), false);
});
test('invalid backend output rejected', async () => {
  const t = manualTimer(), b = controlledBackend(false);
  const p = answerWithDeadline(input, b.backend, { timeoutMs: 10 }, t.schedule);
  b.complete({ ...reply, requestId: 'wrong' }); assert.deepEqual(await p, { ok: false, code: 'INVALID_OUTPUT' });
});
test('settled success cannot be changed by caller or timer', async () => {
  const t = manualTimer(), b = controlledBackend(false), c = new AbortController();
  const p = answerWithDeadline(input, b.backend, { timeoutMs: 10, signal: c.signal }, t.schedule);
  b.complete(reply); await p; c.abort(); t.fire();
  assert.deepEqual(await p, { ok: true, reply }); assert.equal(b.signal()?.aborted, false);
});
test('first observed callback wins, not wall clock speculation', async () => {
  const t = manualTimer(), b = controlledBackend(false);
  const p = answerWithDeadline(input, b.backend, { timeoutMs: 10 }, t.schedule);
  b.complete(reply); t.fire(); // completion handler is still a queued microtask
  assert.deepEqual(await p, { ok: false, code: 'TIMEOUT' });
});
test('real timer smoke test: no exact millisecond assertion', async () => {
  const b = controlledBackend(true);
  assert.deepEqual(await answerWithDeadline(input, b.backend, { timeoutMs: 1 }), { ok: false, code: 'TIMEOUT' });
});
