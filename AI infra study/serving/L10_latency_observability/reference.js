/** Simplest success-only reference: three sequential nonnegative durations. */
export function successReference(queueMs, executionMs, deliveryMs = 0) {
  if ([queueMs, executionMs, deliveryMs].some(x => !Number.isFinite(x) || x < 0))
    throw new RangeError('durations must be finite and nonnegative');
  return { queueWait: queueMs, execution: executionMs,
    callerWait: queueMs + executionMs + deliveryMs };
}
