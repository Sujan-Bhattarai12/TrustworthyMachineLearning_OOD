import unittest
import numpy as np
import pandas as pd
from scripts.evaluate_warning_models import evaluate


class MissingDropoutTests(unittest.TestCase):
    def test_only_selected_baselines_are_evaluated(self):
        rows = []
        for image_id in range(6):
            for corruption in ['gaussian_noise', 'brightness']:
                for severity in range(3):
                    failed = int(severity == 2 and image_id % 2 == 0)
                    for method in ['ig', 'gradcam']:
                        rows.append(dict(source='train',image_id=image_id,corruption=corruption,
                                         severity=severity,clean_class=0,true_label=0,
                                         predicted_class=failed,target_confidence=0.9-severity*0.1,
                                         method=method,valid_map=True,loss=severity*0.1,
                                         gain=severity*0.2,weighted_wasserstein=severity*0.05,
                                         mc_dropout=np.nan,odin=severity*0.01,classifier='synthetic'))
        summary,predictions,excluded,skipped=evaluate(pd.DataFrame(rows))
        self.assertEqual(excluded,0)
        self.assertIn('MSP (raw)',set(summary.model))
        self.assertIn('ODIN (raw)',set(summary.model))
        self.assertNotIn('MC Dropout (raw)',set(summary.model))
        self.assertFalse(skipped)
        self.assertTrue(predictions.groupby(['source','image_id']).fold.nunique().eq(1).all())


if __name__ == '__main__':
    unittest.main()
