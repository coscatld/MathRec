import { answerWithDeadline } from './timeout.ts';
import { manualTimer, controlledBackend } from './fakes.ts';
const timer = manualTimer(), work = controlledBackend(false);
const waiting = answerWithDeadline({ requestId: 'demo', prompt: 'hello', maxOutputChars: 20 }, work.backend, { timeoutMs: 100 }, timer.schedule);
timer.fire();
console.log(JSON.stringify({ result: await waiting, abortRequested: work.signal()?.aborted, workStopped: work.stopped() }));
work.complete({ requestId: 'demo', text: 'late', finishReason: 'stop' });
console.log(JSON.stringify({ workStoppedAfterCompletion: work.stopped() }));
