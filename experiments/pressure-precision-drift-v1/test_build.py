"""Source transformation must preserve arithmetic and distinguish code from strings."""
import unittest
from build import function,specialize


class SourceSpecializationTests(unittest.TestCase):
    def test_extracts_function_with_literal_braces_and_comments(self):
        expected='fn pressure() { let text="}"; /* { */ if true { call(); } }'
        self.assertEqual(function(expected+'\nfn other() {}','fn pressure('),expected)

    def test_only_storage_types_and_casts_change_not_policy_constants(self):
        source='let old: &[f32] = input; let stored = proposed as f32; let cap = f64::from(f32::MAX);'
        self.assertEqual(specialize(source),
            'let old: &[f64] = input; let stored = proposed as f64; let cap = f64::from(f32::MAX);')


if __name__=='__main__':unittest.main()
