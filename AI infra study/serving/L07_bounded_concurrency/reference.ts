/** Small sequential reference for successful mapping, not a queue implementation. */
export async function sequentialMap<T, R>(items: readonly T[], work: (item: T) => Promise<R>): Promise<R[]> {
  const results: R[] = [];
  for (const item of items) results.push(await work(item));
  return results;
}
