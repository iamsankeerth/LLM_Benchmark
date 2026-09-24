import json
import unittest
from pathlib import Path
from candidate import merge_intervals

class TestQ029(unittest.TestCase):
    def test_cases(self):
        cases=json.loads((Path(__file__).with_name('cases.json')).read_text())
        for case in cases:
            with self.subTest(case['id']):
                self.assertEqual(merge_intervals(case['input']), case['expected'])
