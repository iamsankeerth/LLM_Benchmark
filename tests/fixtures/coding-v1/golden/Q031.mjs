export function groupBy(items) {
  const result = Object.create(null);
  for (const item of items) {
    const key = Object.prototype.hasOwnProperty.call(item, 'kind') ? item.kind : '__missing__';
    if (!Object.prototype.hasOwnProperty.call(result, key)) result[key] = [];
    result[key].push(item);
  }
  return result;
}
