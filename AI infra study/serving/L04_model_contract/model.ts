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
export interface ModelBackend {
  generate(request: Request): Promise<unknown>;
}
export type Result =
  | { ok: true; reply: Reply }
  | { ok: false; code: 'INVALID_INPUT' | 'BACKEND_FAILED' | 'INVALID_OUTPUT' };

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
export async function answer(input: unknown, backend: ModelBackend): Promise<Result> {
  const request = parseRequest(input);
  if (request === null) return { ok: false, code: 'INVALID_INPUT' };
  let raw: unknown;
  try {
    raw = await backend.generate(request);
  } catch {
    // Deliberately omit raw provider exceptions from public output.
    return { ok: false, code: 'BACKEND_FAILED' };
  }
  const reply = parseReply(raw, request);
  if (reply === null) return { ok: false, code: 'INVALID_OUTPUT' };
  return { ok: true, reply };
}

// Deterministic mock; no inference, network, randomness, clock, or shared session.
export const mockBackend: ModelBackend = {
  async generate(request: Request): Promise<unknown> {
    const full = `mock:${request.prompt}`;
    return {
      requestId: request.requestId,
      text: full.slice(0, request.maxOutputChars),
      finishReason: full.length > request.maxOutputChars ? 'length' : 'stop',
    };
  },
};
