import json
import unittest
from pathlib import Path
from candidate import top_k_frequent

class TestQ033(unittest.TestCase):
    def test_cases(self):
        cases=json.loads((Path(__file__).with_name('cases.json')).read_text())
        for case in cases:
            with self.subTest(case['id']):
                self.assertEqual(top_k_frequent(case['input'], case['k']), case['expected'])
