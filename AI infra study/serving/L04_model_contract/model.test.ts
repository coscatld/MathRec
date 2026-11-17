import test from 'node:test';
import assert from 'node:assert/strict';
import { answer, mockBackend, parseRequest } from './model.ts';
import type { ModelBackend, Request } from './model.ts';
const input = () => ({ requestId: 'r-1', prompt: 'hello', maxOutputChars: 64 });
const output = (overrides: Record<string, unknown> = {}) => ({ requestId: 'r-1', text: 'ok', finishReason: 'stop', ...overrides });

test('deterministic mock and complete normalized output', async () => {
  const a = await answer(input(), mockBackend);
  assert.deepEqual(a, { ok: true, reply: output({ text: 'mock:hello' }) });
  assert.deepEqual(a, await answer(input(), mockBackend));
});
test('output limit uses UTF-16 code units and length finish reason', async () => {
  assert.deepEqual(await answer({ ...input(), maxOutputChars: 4 }, mockBackend), {
    ok: true, reply: output({ text: 'mock', finishReason: 'length' }),
  });
});
test('exact boundary stops, not length', async () => {
  const r = await answer({ ...input(), maxOutputChars: 10 }, mockBackend);
  assert.equal(r.ok && r.reply.finishReason, 'stop');
});
test('invalid inputs never call backend', async () => {
  let calls = 0;
  const backend: ModelBackend = { async generate() { calls++; return output(); } };
  const bad: unknown[] = [null, [], {}, { ...input(), requestId: '' }, { ...input(), requestId: 'x'.repeat(65) },
    { ...input(), requestId: 'bad id' }, { ...input(), prompt: '  ' }, { ...input(), prompt: 1 },
    { ...input(), prompt: 'x'.repeat(4097) }, { ...input(), maxOutputChars: 0 },
    { ...input(), maxOutputChars: 513 }, { ...input(), maxOutputChars: 1.5 },
    { ...input(), maxOutputChars: NaN }, { ...input(), maxOutputChars: Infinity },
    { ...input(), maxOutputChars: '64' }];
  for (const x of bad) assert.deepEqual(await answer(x, backend), { ok: false, code: 'INVALID_INPUT' });
  assert.equal(calls, 0);
});
test('maximum valid input and minimum output bound', async () => {
  assert.ok(parseRequest({ requestId: 'x'.repeat(64), prompt: 'x'.repeat(4096), maxOutputChars: 512 }));
  const r = await answer({ ...input(), maxOutputChars: 1 }, mockBackend);
  assert.equal(r.ok && r.reply.text, 'm');
});
test('malformed outputs fail closed', async () => {
  const bad: unknown[] = [null, [], {}, output({ requestId: 'other' }), output({ text: 7 }),
    output({ text: 'x'.repeat(65) }), output({ finishReason: 'unknown' })];
  for (const value of bad) {
    assert.deepEqual(await answer(input(), { async generate() { return value; } }),
      { ok: false, code: 'INVALID_OUTPUT' });
  }
});
test('valid empty answer allowed; extra fields dropped', async () => {
  const r = await answer(input(), { async generate() { return output({ text: '', debug: 'not public' }); } });
  assert.deepEqual(r, { ok: true, reply: output({ text: '' }) });
});
test('rejection is normalized without provider text', async () => {
  const result = await answer(input(), { async generate() { throw new Error('internal detail'); } });
  assert.deepEqual(result, { ok: false, code: 'BACKEND_FAILED' });
});
test('synchronous throw is also normalized', async () => {
  const backend: ModelBackend = { generate() { throw new Error('sync'); } };
  assert.deepEqual(await answer(input(), backend), { ok: false, code: 'BACKEND_FAILED' });
});
test('request snapshot cannot follow caller mutation while pending', async () => {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  let seen: Request | undefined;
  const backend: ModelBackend = { async generate(r) {
    seen = r; await gate; return { requestId: r.requestId, text: r.prompt, finishReason: 'stop' };
  } };
  const original = { ...input(), ignored: 'discard' };
  const pending = answer(original, backend);
  original.prompt = 'changed'; original.requestId = 'changed';
  assert.equal(Object.isFrozen(seen), true);
  assert.deepEqual(Object.keys(seen!).sort(), ['maxOutputChars', 'prompt', 'requestId']);
  release();
  assert.deepEqual(await pending, { ok: true, reply: output({ text: 'hello' }) });
});
test('independent simultaneous requests do not share request state', async () => {
  const results = await Promise.all(Array.from({ length: 30 }, (_, i) =>
    answer({ requestId: `r-${i}`, prompt: String(i), maxOutputChars: 64 }, mockBackend)));
  results.forEach((r, i) => assert.deepEqual(r, {
    ok: true, reply: { requestId: `r-${i}`, text: `mock:${i}`, finishReason: 'stop' },
  }));
});
test('function injection swaps behavior without changing answer', async () => {
  const backend: ModelBackend = { async generate(r) {
    return { requestId: r.requestId, text: 'replacement', finishReason: 'stop' };
  } };
  const r = await answer(input(), backend);
  assert.equal(r.ok && r.reply.text, 'replacement');
});
test('UTF-16 budget can split an emoji surrogate pair: documented teaching limit', async () => {
  const r = await answer({ ...input(), prompt: '😀', maxOutputChars: 6 }, mockBackend);
  assert.equal(r.ok, true);
  if (!r.ok) throw new Error('expected success');
  assert.equal(r.reply.text.length, 6);
  assert.equal(r.reply.text.charCodeAt(5), 0xD83D);
  assert.equal(r.reply.text.isWellFormed(), false);
  assert.equal(r.reply.finishReason, 'length');
});
