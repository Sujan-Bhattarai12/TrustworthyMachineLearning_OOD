import unittest
from src.warning_features import validate_attribution_columns


class AttributionOnlyTests(unittest.TestCase):
    def test_guard_rejects_confidence_and_labels(self):
        for col in ['unconfidence', 'msp', 'odin', 'delta_unconfidence',
                    'fails_next_step', 'first_failure', 'severity']:
            with self.subTest(column=col):
                with self.assertRaises(ValueError):
                    validate_attribution_columns(['ig_loss', col])


if __name__ == '__main__':
    unittest.main()
