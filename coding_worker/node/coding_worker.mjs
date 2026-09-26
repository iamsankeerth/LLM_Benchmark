import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

function truncate(value, limit) {
  const buffer = Buffer.from(value);
  return buffer.length <= limit ? value : buffer.subarray(0, limit).toString('utf8');
}

function emit(value) {
  process.stdout.write(`${JSON.stringify(value)}\n`);
}

function failureKind(error) {
  if (error && error.code === 'ETIMEDOUT') return 'TIMEOUT';
  return 'WORKER_ERROR';
}

let request;
try {
  request = JSON.parse(fs.readFileSync(0, 'utf8'));
  if (request.language !== 'javascript') throw new Error('node worker received another language');
  if (Buffer.byteLength(request.candidate_source, 'utf8') > Number(request.policy.candidate_source_bytes)) {
    throw new Error('candidate source exceeds policy');
  }
} catch (error) {
  emit({
    passed: false,
    failure_kind: 'WORKER_INPUT_ERROR',
    error: String(error),
    tests: { static_passed: 0, total: 0 },
  });
  process.exit(0);
}

const work = fs.mkdtempSync(path.join(os.tmpdir(), 'coding-'));
try {
  fs.writeFileSync(path.join(work, 'candidate.mjs'), request.candidate_source, 'utf8');
  fs.writeFileSync(path.join(work, 'test_candidate.mjs'), request.test_source, 'utf8');
  fs.writeFileSync(path.join(work, 'cases.json'), request.cases_json, 'utf8');
  const completed = spawnSync(
    process.execPath,
    ['--test', 'test_candidate.mjs'],
    {
      cwd: work,
      encoding: 'utf8',
      timeout: Number(request.policy.whole_candidate_wall_seconds) * 1000,
      env: { ...process.env, HOME: os.tmpdir() },
      maxBuffer: Number(request.policy.output_bytes),
    },
  );
  const stdout = completed.stdout || '';
  const stderr = completed.stderr || '';
  const totalMatch = stdout.match(/^# tests (\d+)$/m);
  const passMatch = stdout.match(/^# pass (\d+)$/m);
  const totalTests = totalMatch ? Number(totalMatch[1]) : 0;
  const passedTests = passMatch ? Number(passMatch[1]) : 0;
  const passed = completed.status === 0 && totalTests > 0 && passedTests === totalTests;
  emit({
    passed,
    failure_kind: passed ? null : failureKind(completed.error),
    stdout: truncate(stdout, Number(request.policy.output_bytes)),
    stderr: truncate(stderr, Number(request.policy.output_bytes)),
    tests: {
      static_passed: passed ? totalTests : passedTests,
      total: totalTests,
    },
  });
} catch (error) {
  emit({
    passed: false,
    failure_kind: 'WORKER_ERROR',
    error: String(error),
    tests: { static_passed: 0, total: 0 },
  });
} finally {
  fs.rmSync(work, { recursive: true, force: true });
}
