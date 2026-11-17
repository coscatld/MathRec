import { performance } from 'node:perf_hooks';
import { answer, mockBackend } from './model.ts';
const request = { requestId: 'bench', prompt: 'hello', maxOutputChars: 64 };
for (let i = 0; i < 1000; i++) await answer(request, mockBackend);
const batches: number[] = [];
const perBatch = 10000;
for (let b = 0; b < 9; b++) {
  const start = performance.now();
  for (let i = 0; i < perBatch; i++) {
    const result = await answer(request, mockBackend);
    if (!result.ok) throw new Error('benchmark correctness failure');
  }
  batches.push((performance.now() - start) * 1000 / perBatch);
}
batches.sort((a, b) => a - b);
console.log(JSON.stringify({ node: process.version, samples: batches.length, callsPerSample: perBatch,
  medianBatchMeanUs: batches[4], minBatchMeanUs: batches[0], maxBatchMeanUs: batches[8],
  scope: 'sequential local mock + validation + Promise overhead; NOT model/network latency or request p95' }, null, 2));
