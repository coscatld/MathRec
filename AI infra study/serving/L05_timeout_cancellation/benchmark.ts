import { performance } from 'node:perf_hooks';
import { answerWithDeadline, type CancellableBackend } from './timeout.ts';
const backend: CancellableBackend = { async generate(r) { return { requestId: r.requestId, text: 'ok', finishReason: 'stop' }; } };
const input = { requestId: 'bench', prompt: 'hello', maxOutputChars: 20 };
const run = async () => { if (!(await answerWithDeadline(input, backend, { timeoutMs: 1000 })).ok) throw new Error('unexpected failure'); };
for (let i=0; i<1000; i++) await run();
const samples: number[]=[];
for (let b=0; b<7; b++) { const start=performance.now(); for (let i=0;i<5000;i++) await run(); samples.push((performance.now()-start)*1000/5000); }
samples.sort((a,b)=>a-b);
console.log(JSON.stringify({ node:process.version, batches:7, callsPerBatch:5000, medianBatchMeanUs:samples[3], scope:'local immediate mock + timer setup/cleanup; NOT request p95 or real model latency' },null,2));
