import { performance } from 'node:perf_hooks';
import { Recorder } from './metrics.js';
const N = 10000;
function run() {
  let t = 0;
  const r = new Recorder(() => t++);
  const begin = performance.now();
  for (let i = 0; i < N; i++) { const id = String(i); r.accept(id); r.start(id); r.finishBackend(id, 'ok'); r.finishCaller(id, 'ok'); }
  const recorded = performance.now();
  r.report();
  return { recordMs: recorded - begin, reportMs: performance.now() - recorded };
}
run(); // warmup, not reported
console.log(JSON.stringify({ node: process.version, requests: N, trials: Array.from({length:5}, run),
  scope: 'Local recorder overhead only; fake durations, no server/model/GPU throughput measured.' }, null, 2));
