import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.attribution_features import add_features
from scripts.evaluate_individual_lgd import recipes,fit_components

class IndividualLgdAblations(unittest.TestCase):
    def test_six_single_component_recipes(self):
        r=recipes();self.assertEqual(len(r),6)
        self.assertEqual({s['label'] for s in r},set('LGD'))
        self.assertTrue(all(not s['column'].startswith('delta_') for s in r))
        self.assertTrue(all('msp' not in s['column'] and 'confidence' not in s['column'] for s in r))
    def test_fit_uses_original_groups_and_one_feature(self):
        root=Path('results/expanded_cifar10c_5000dev_2000test')
        data,_=add_features(pd.read_csv(root/'development/resnet50/images.csv'))
        artifact=json.loads((root/'frozen/resnet50/specifications.json').read_text())
        models,specs=fit_components(data,artifact)
        self.assertEqual(len(models),6)
        self.assertTrue(all(m.n_features_in_==1 for m in models.values()))
        self.assertTrue(all(np.isfinite(s['threshold']) and s['calibration_fpr']<=.05 for s in specs))
        overlapping=dict(artifact,calibration_image_groups=artifact['fit_image_groups'])
        with self.assertRaises(ValueError):fit_components(data,overlapping)
