import test from 'node:test';
import assert from 'node:assert/strict';
import cases from './cases.json' with { type: 'json' };
import { groupBy } from './candidate.mjs';

test('Q031 cases', () => {
  for (const item of cases) {
    const result = groupBy(item.input);
    for (const [key, values] of Object.entries(item.expected)) {
      assert.deepEqual(result[key], values);
    }
  }
});
