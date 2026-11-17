import assert from 'node:assert/strict';
import { setTimeout as delay } from 'node:timers/promises';
import { BoundedExecutor } from './executor.ts';
const count = 24;
const delayMs = 3;
const repetitions = 7;
async function run(concurrency: number) {
  const pool = new BoundedExecutor<number>({ concurrency, maxQueue: count });
  const begin = performance.now();
  const results = await Promise.all(Array.from({ length: count }, (_, i) =>
    pool.submit(async (signal) => { await delay(delayMs, undefined, { signal }); return i; }, { timeoutMs: 60000 })));
  const totalMs = performance.now() - begin;
  assert.deepEqual(results.map(r => r.ok ? r.value : undefined), Array.from({ length: count }, (_, i) => i));
  assert.equal(pool.snapshot().active, 0);
  assert.ok(pool.snapshot().peakActive <= concurrency);
  return totalMs;
}
// Correctness and one warmup per configuration before collecting measurements.
await run(1); await run(4);
const samples: Record<string, number[]> = { C1: [], C4: [] };
for (let i = 0; i < repetitions; i++) {
  for (const c of i % 2 === 0 ? [1, 4] : [4, 1]) samples[`C${c}`].push(await run(c));
}
for (const values of Object.values(samples)) values.sort((a, b) => a - b);
console.log(JSON.stringify({ node: process.version, platform: `${process.platform}/${process.arch}`,
  count, delayMs, repetitions, unit: 'whole batch wall-clock milliseconds',
  scope: 'local timer mock; no network, model, CPU parallelism or service latency claim',
  results: Object.fromEntries(Object.entries(samples).map(([key, ms]) => [key, {
    medianMs: ms[Math.floor(ms.length / 2)], minMs: ms[0], maxMs: ms.at(-1), samplesMs: ms,
  }])),
}, null, 2));
