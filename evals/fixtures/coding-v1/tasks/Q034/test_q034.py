import json
import unittest
from pathlib import Path
from candidate import run_with_retry

class TestQ034(unittest.TestCase):
    def test_cases(self):
        cases=json.loads((Path(__file__).with_name('cases.json')).read_text())
        for case in cases:
            with self.subTest(case['id']):
                calls=[0]
                def operation():
                    calls[0]+=1
                    if calls[0] <= case['failures']:
                        raise RuntimeError('retry')
                    return 'ok'
                if case['failures'] >= 3:
                    with self.assertRaises(RuntimeError):
                        run_with_retry(operation)
                else:
                    self.assertEqual(run_with_retry(operation), 'ok')
                self.assertEqual(calls[0], case['expected_calls'])
