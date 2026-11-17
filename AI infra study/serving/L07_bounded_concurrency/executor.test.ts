import test from 'node:test';
import assert from 'node:assert/strict';
import { getEventListeners } from 'node:events';
import { BoundedExecutor } from './executor.ts';
import { ControlledBackend, ManualClock, flush } from './fakes.ts';
import { sequentialMap } from './reference.ts';
function setup(concurrency = 1, maxQueue = 2) {
  const clock = new ManualClock();
  const backend = new ControlledBackend();
  const pool = new BoundedExecutor<string>({ concurrency, maxQueue }, clock);
  const submit = (id: string, timeoutMs = 100, signal?: AbortSignal) =>
    pool.submit((owned) => backend.generate(id, owned), { timeoutMs, signal });
  return { clock, backend, pool, submit };
}

test('configuration rejects invalid integer bounds', () => {
  for (const concurrency of [0, -1, 1.5, NaN, Infinity, 1025, Number.MAX_SAFE_INTEGER + 1])
    assert.throws(() => new BoundedExecutor({ concurrency, maxQueue: 1 }), RangeError);
  for (const maxQueue of [-1, 1.5, NaN, Infinity, 100001])
    assert.throws(() => new BoundedExecutor({ concurrency: 1, maxQueue }), RangeError);
});
test('invalid timeout never starts work or allocates timer', async () => {
  const s = setup();
  for (const timeoutMs of [0, -1, 0.5, NaN, Infinity, 60001])
    assert.deepEqual(await s.submit('bad', timeoutMs), { ok: false, code: 'INVALID_OPTIONS' });
  assert.equal(s.backend.calls.length, 0);
  assert.equal(s.clock.pending, 0);
});
test('pre-aborted signal wins without admission', async () => {
  const s = setup(); const controller = new AbortController(); controller.abort();
  assert.deepEqual(await s.submit('a', 100, controller.signal), { ok: false, code: 'CANCELLED' });
  assert.equal(s.pool.snapshot().started, 0);
  assert.equal(s.clock.pending, 0);
});
test('C running, Q waiting, next rejected before backend', async () => {
  const s = setup(2, 2);
  const accepted = ['a', 'b', 'c', 'd'].map(x => s.submit(x));
  assert.deepEqual(await s.submit('e'), { ok: false, code: 'OVERLOADED' });
  assert.equal(s.pool.snapshot().active, 2); assert.equal(s.pool.snapshot().queued, 2);
  assert.equal(s.clock.pending, 4); assert.equal(s.backend.calls.length, 2);
  s.backend.calls[0].finish(); s.backend.calls[1].finish(); await flush();
  assert.deepEqual(s.backend.calls.map(x => x.id), ['a', 'b', 'c', 'd']);
  s.backend.calls[2].finish(); s.backend.calls[3].finish();
  assert.ok((await Promise.all(accepted)).every(x => x.ok));
  assert.equal(s.backend.peak, 2); assert.equal(s.clock.pending, 0);
});
test('Q=0 allows free slot and rejects overflow', async () => {
  const s = setup(1, 0); const a = s.submit('a');
  assert.deepEqual(await s.submit('b'), { ok: false, code: 'OVERLOADED' });
  s.backend.calls[0].finish(); await a;
  const c = s.submit('c'); s.backend.calls[1].finish(); await c;
  assert.equal(s.pool.snapshot().peakQueued, 0);
});
test('FIFO start order differs from completion order', async () => {
  const s = setup(2, 2); const promises = ['a', 'b', 'c', 'd'].map(x => s.submit(x));
  s.backend.calls[1].finish(); await flush();
  assert.deepEqual(s.backend.calls.map(x => x.id), ['a', 'b', 'c']);
  s.backend.calls[2].finish(); await flush();
  assert.deepEqual(s.backend.calls.map(x => x.id), ['a', 'b', 'c', 'd']);
  s.backend.calls[0].finish(); s.backend.calls[3].finish(); await Promise.all(promises);
});
test('queued timeout includes wait and immediately frees Q', async () => {
  const s = setup(1, 1); const a = s.submit('a', 100); const b = s.submit('b', 10);
  s.clock.advance(10); assert.deepEqual(await b, { ok: false, code: 'TIMEOUT' });
  assert.equal(s.pool.snapshot().queued, 0);
  const c = s.submit('c'); s.backend.calls[0].finish(); await flush();
  assert.deepEqual(s.backend.calls.map(x => x.id), ['a', 'c']);
  s.backend.calls[1].finish(); await Promise.all([a, c]);
});
test('queued cancellation removes entry and abort listener', async () => {
  const s = setup(); const a = s.submit('a'); const ctrl = new AbortController();
  const b = s.submit('b', 100, ctrl.signal); const c = s.submit('c');
  ctrl.abort(); assert.deepEqual(await b, { ok: false, code: 'CANCELLED' });
  assert.equal(s.pool.snapshot().queued, 1);
  assert.equal(getEventListeners(ctrl.signal, 'abort').length, 0);
  s.backend.calls[0].finish(); await flush();
  assert.deepEqual(s.backend.calls.map(x => x.id), ['a', 'c']);
  s.backend.calls[1].finish(); await Promise.all([a, c]);
});
test('running timeout keeps slot until ignored-abort backend finishes', async () => {
  const s = setup(); const a = s.submit('a', 10); const b = s.submit('b', 100);
  s.clock.advance(10); assert.deepEqual(await a, { ok: false, code: 'TIMEOUT' });
  assert.equal(s.backend.calls[0].aborted, true);
  assert.equal(s.pool.snapshot().active, 1); assert.equal(s.backend.active, 1);
  assert.equal(s.backend.calls.length, 1);
  s.backend.calls[0].finish('late success'); await flush();
  assert.equal(s.backend.calls.length, 2); assert.equal(s.backend.peak, 1);
  s.backend.calls[1].finish(); assert.equal((await b).ok, true);
  assert.deepEqual(await a, { ok: false, code: 'TIMEOUT' });
});
test('running cancellation likewise retains capacity', async () => {
  const s = setup(); const ctrl = new AbortController(); const a = s.submit('a', 100, ctrl.signal);
  const b = s.submit('b'); ctrl.abort();
  assert.deepEqual(await a, { ok: false, code: 'CANCELLED' });
  assert.equal(s.pool.snapshot().active, 1); assert.equal(s.backend.calls.length, 1);
  s.backend.calls[0].finish(); await flush(); s.backend.calls[1].finish(); await b;
});
test('late rejection is observed and slot released exactly once', async () => {
  const s = setup(); const a = s.submit('a', 10);
  s.clock.advance(10); await a; s.backend.calls[0].fail(); await flush();
  assert.equal(s.pool.snapshot().active, 0); assert.equal(s.pool.snapshot().completed, 1);
  s.backend.calls[0].finish(); await flush(); assert.equal(s.pool.snapshot().completed, 1);
  assert.deepEqual(await a, { ok: false, code: 'TIMEOUT' });
});
test('cooperative cancellation settles result before backend rejection', async () => {
  const s = setup();
  const a = s.pool.submit(signal => s.backend.generate('a', signal, true), { timeoutMs: 10 });
  s.clock.advance(10); assert.deepEqual(await a, { ok: false, code: 'TIMEOUT' });
  await flush(); assert.equal(s.pool.snapshot().active, 0); assert.equal(s.clock.pending, 0);
});
test('sync throw, async rejection and success all release capacity', async () => {
  const s = setup();
  const a = s.pool.submit(() => { throw Error('sync'); }, { timeoutMs: 100 });
  const b = s.pool.submit(async () => { throw Error('async'); }, { timeoutMs: 100 });
  const c = s.pool.submit(() => 'ok', { timeoutMs: 100 });
  assert.deepEqual(await a, { ok: false, code: 'TASK_FAILED' });
  assert.deepEqual(await b, { ok: false, code: 'TASK_FAILED' });
  assert.deepEqual(await c, { ok: true, value: 'ok' });
  assert.equal(s.pool.snapshot().active, 0); assert.equal(s.clock.pending, 0);
});
test('normal completion clears timer and caller listener', async () => {
  const s = setup(); const ctrl = new AbortController(); const a = s.submit('a', 100, ctrl.signal);
  assert.equal(getEventListeners(ctrl.signal, 'abort').length, 1);
  s.backend.calls[0].finish(); assert.equal((await a).ok, true);
  assert.equal(s.clock.pending, 0); assert.equal(getEventListeners(ctrl.signal, 'abort').length, 0);
  ctrl.abort(); s.clock.advance(1000); assert.equal(s.pool.snapshot().completed, 1);
});
test('expired queued work never starts when timer callback is delayed', async () => {
  const s = setup(); const a = s.submit('a', 100); const b = s.submit('b', 10); const c = s.submit('c', 100);
  s.clock.elapseWithoutTimers(10); s.backend.calls[0].finish(); await flush();
  assert.deepEqual(await b, { ok: false, code: 'TIMEOUT' });
  assert.deepEqual(s.backend.calls.map(x => x.id), ['a', 'c']);
  s.backend.calls[1].finish(); await Promise.all([a, c]); assert.equal(s.clock.pending, 0);
});
test('completion at deadline is timeout even before timer callback', async () => {
  const s = setup(); const a = s.submit('a', 10);
  s.clock.elapseWithoutTimers(10); s.backend.calls[0].finish();
  assert.deepEqual(await a, { ok: false, code: 'TIMEOUT' });
  assert.equal(s.pool.snapshot().active, 0); assert.equal(s.clock.pending, 0);
});
test('completion before deadline succeeds', async () => {
  const s = setup(); const a = s.submit('a', 10);
  s.clock.elapseWithoutTimers(9); s.backend.calls[0].finish();
  assert.deepEqual(await a, { ok: true, value: 'reply:a' });
});
test('first public reason survives later timeout or cancellation', async () => {
  const s = setup(); const ctrl = new AbortController(); const a = s.submit('a', 10, ctrl.signal);
  ctrl.abort(); s.clock.advance(20); s.backend.calls[0].fail(); await flush();
  assert.deepEqual(await a, { ok: false, code: 'CANCELLED' });
});
test('reentrant submit cannot exceed capacity', async () => {
  const s = setup(1, 1); let nested: ReturnType<typeof s.submit> | undefined;
  const a = s.pool.submit(signal => { nested = s.submit('b'); return s.backend.generate('a', signal); }, { timeoutMs: 100 });
  assert.equal(s.backend.calls.length, 1); assert.equal(s.pool.snapshot().queued, 1);
  s.backend.calls[0].finish(); await flush(); s.backend.calls[1].finish(); await Promise.all([a, nested]);
  assert.equal(s.backend.peak, 1);
});
test('stress: 1000 submissions enforce C=3, Q=7, bounded admission', async () => {
  const s = setup(3, 7); const promises = Array.from({ length: 1000 }, (_, i) => s.submit(String(i)));
  assert.equal(s.pool.snapshot().active, 3); assert.equal(s.pool.snapshot().queued, 7);
  assert.equal(s.pool.snapshot().overloaded, 990); assert.equal(s.clock.pending, 10);
  let finished = 0;
  while (finished < 10) { s.backend.calls[finished++].finish(); await flush(); }
  const results = await Promise.all(promises);
  assert.equal(results.filter(x => x.ok).length, 10); assert.equal(s.backend.peak, 3);
  assert.equal(s.pool.snapshot().peakQueued, 7); assert.equal(s.clock.pending, 0);
});
test('sequential reference agrees on successful result ordering', async () => {
  const values = [1, 2, 3, 4];
  const expected = await sequentialMap(values, async x => x * 2);
  const pool = new BoundedExecutor<number>({ concurrency: 2, maxQueue: 2 });
  const results = await Promise.all(values.map(x => pool.submit(async () => x * 2, { timeoutMs: 1000 })));
  assert.deepEqual(results.map(x => x.ok ? x.value : undefined), expected);
});
test('real timer smoke returns timeout, but does not prove exact latency', async () => {
  const pool = new BoundedExecutor<string>({ concurrency: 1, maxQueue: 0 });
  const backend = new ControlledBackend();
  const result = await pool.submit(signal => backend.generate('a', signal), { timeoutMs: 1 });
  assert.deepEqual(result, { ok: false, code: 'TIMEOUT' });
  assert.equal(pool.snapshot().active, 1); backend.calls[0].finish(); await flush();
  assert.equal(pool.snapshot().active, 0);
});
