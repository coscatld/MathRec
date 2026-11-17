/** Independent deterministic audit. Run: node audit_stress.ts. No dependencies. */
import assert from 'node:assert/strict';
import { BoundedExecutor } from './executor.ts';
import type { Outcome } from './executor.ts';
import { ManualClock, ControlledBackend, flush } from './fakes.ts';

for (let seed = 1; seed <= 200; seed++) {
  let state = seed;
  const random = () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 2 ** 32);
  const concurrency = 1 + (seed - 1) % 4;
  const maxQueue = Math.floor((seed - 1) / 4) % 5; // all 20 C/Q pairs, ten times each
  const clock = new ManualClock();
  const backend = new ControlledBackend();
  const executor = new BoundedExecutor<string>({ concurrency, maxQueue }, clock);
  const controllers: AbortController[] = [];
  const outcomes: Promise<Outcome<string>>[] = [];
  let id = 0;
  for (let step = 0; step < 100; step++) {
    const operation = Math.floor(random() * 4);
    if (operation === 0) {
      const controller = new AbortController();
      controllers.push(controller);
      outcomes.push(executor.submit(
        signal => backend.generate(String(id++), signal, random() < 0.5),
        { timeoutMs: 1 + Math.floor(random() * 20), signal: controller.signal },
      ));
    }
    if (operation === 1 && controllers.length)
      controllers[Math.floor(random() * controllers.length)].abort();
    if (operation === 2 && backend.calls.length) {
      const call = backend.calls[Math.floor(random() * backend.calls.length)];
      if (random() < 0.5) call.finish();
      else call.fail();
    }
    if (operation === 3) clock.advance(Math.floor(random() * 10));
    await flush();
    const snapshot = executor.snapshot();
    const context = `seed=${seed}, step=${step}`;
    assert(snapshot.active <= concurrency, context);
    assert(snapshot.queued <= maxQueue, context);
    assert.equal(snapshot.active, backend.active, context);
  }
  controllers.forEach(controller => controller.abort());
  backend.calls.forEach(call => call.finish());
  await flush();
  clock.advance(100);
  await Promise.all(outcomes);
  assert.equal(executor.snapshot().active, 0, `seed=${seed}`);
  assert.equal(executor.snapshot().queued, 0, `seed=${seed}`);
  assert.equal(clock.pending, 0, `seed=${seed}`);
}
console.log('Independent randomized smoke: 200 seeds × 100 operations passed');
