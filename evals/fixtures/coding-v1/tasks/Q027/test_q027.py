import json
import unittest
from pathlib import Path
from candidate import dedupe_preserve_order

class TestQ027(unittest.TestCase):
    def test_cases(self):
        cases=json.loads((Path(__file__).with_name('cases.json')).read_text())
        for case in cases:
            with self.subTest(case['id']):
                self.assertEqual(dedupe_preserve_order(case['input']), case['expected'])
