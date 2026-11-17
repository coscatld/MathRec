import { answer, mockBackend } from './model.ts';
const result = await answer({ requestId: 'demo-1', prompt: 'hello', maxOutputChars: 64 }, mockBackend);
console.log(JSON.stringify(result, null, 2));
