// Teaching boundary: JSON-shaped values, not hostile in-process Proxies/getters.
export type Request = Readonly<{
  requestId: string;
  prompt: string;
  maxOutputChars: number;
}>;
export type Reply = Readonly<{
  requestId: string;
  text: string;
  finishReason: 'stop' | 'length';
}>;
function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
export function parseRequest(value: unknown): Request | null {
  if (!record(value)) return null;
  const { requestId, prompt, maxOutputChars } = value;
  if (typeof requestId !== 'string' || !/^[A-Za-z0-9_-]{1,64}$/.test(requestId)) return null;
  if (typeof prompt !== 'string' || prompt.trim().length === 0 || prompt.length > 4096) return null;
  if (typeof maxOutputChars !== 'number' || !Number.isInteger(maxOutputChars)
      || maxOutputChars < 1 || maxOutputChars > 512) return null;
  // Copy accepted fields before any await. Unknown fields are intentionally dropped.
  return Object.freeze({ requestId, prompt, maxOutputChars });
}
export function parseReply(value: unknown, request: Request): Reply | null {
  if (!record(value)) return null;
  const { requestId, text, finishReason } = value;
  if (requestId !== request.requestId || typeof text !== 'string') return null;
  if (text.length > request.maxOutputChars) return null;
  if (finishReason !== 'stop' && finishReason !== 'length') return null;
  return Object.freeze({ requestId: request.requestId, text, finishReason });
}
