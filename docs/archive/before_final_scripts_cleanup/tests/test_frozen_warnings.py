import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.freeze_warning_methods import fit_spec,predict
from scripts.archive.legacy_entrypoints.evaluate_frozen_warnings import evaluate


class FrozenWarningTests(unittest.TestCase):
    def test_fixed_scales_only_depend_on_fit_data(self):
        fit=pd.DataFrame(dict(eligible_fit=[True,True],gradcam_loss=[1.,2.],gradcam_weighted_wasserstein=[2.,4.]))
        spec=fit_spec('GRADCAM L+D fixed',fit)
        evaluation=pd.DataFrame(dict(gradcam_loss=[5.],gradcam_weighted_wasserstein=[10.]))
        before=json.dumps(spec,sort_keys=True)
        expected=(5/spec['scale'][0]+10/spec['scale'][1])/2
        self.assertAlmostEqual(float(predict(spec,evaluation)[0]),expected)
        self.assertEqual(before,json.dumps(spec,sort_keys=True))

    def test_serialized_logistic_score_matches_formula(self):
        data=pd.DataFrame(dict(ig_loss=[1.,2.],ig_gain=[0.,1.],ig_weighted_wasserstein=[2.,3.],eligible_fit=[True,True],fails_next_step=[0,1]))
        spec=fit_spec('IG L+G+D learned',data)
        restored=json.loads(json.dumps(spec))
        np.testing.assert_allclose(predict(spec,data),predict(restored,data))
        self.assertTrue((predict(restored,data)>0).all())

    def test_final_evaluator_rejects_development_data_before_any_fitting(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            (p/'artifact.json').write_text(json.dumps(dict(classifier='resnet50')))
            pd.DataFrame(dict(source=['train'],corruption=['brightness'],classifier=['resnet50'])).to_csv(p/'input.csv',index=False)
            with patch('sklearn.preprocessing.StandardScaler.fit',side_effect=AssertionError('No fitting permitted')):
                with self.assertRaisesRegex(ValueError,'reserved test-source blur'):
                    evaluate(p/'artifact.json',p/'input.csv',p/'out')


if __name__=='__main__':
    unittest.main()
