import { Recorder } from './metrics.js';
export function replay() {
  let now = 0;
  const recorder = new Recorder(() => now);
  const at = (t, action) => { now = t; action(); };
  // A runs slowly; B leaves while queued; C starts at zero and succeeds.
  at(0, () => { recorder.accept('A'); recorder.accept('B'); recorder.accept('C'); recorder.start('C'); });
  at(10, () => recorder.start('A'));
  at(20, () => recorder.finishCaller('B', 'timeout'));
  at(30, () => { recorder.finishBackend('C', 'ok'); recorder.finishCaller('C', 'ok'); });
  at(40, () => recorder.finishCaller('A', 'timeout'));
  const afterTimeout = recorder.report(); // A still occupies a running slot.
  at(100, () => recorder.finishBackend('A', 'ok'));
  return { afterTimeout, final: recorder.report(), traces: recorder.snapshot() };
}
