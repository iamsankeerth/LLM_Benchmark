import json
import unittest
from pathlib import Path
from candidate import is_palindrome

class TestQ030(unittest.TestCase):
    def test_cases(self):
        cases=json.loads((Path(__file__).with_name('cases.json')).read_text())
        for case in cases:
            with self.subTest(case['id']):
                self.assertEqual(is_palindrome(case['input']), case['expected'])
