import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scripts.archive.fresh_trend_validation import fit_model,evaluate_fresh,sha
from src.data_reservation import reserve_fresh
from scripts.freeze_warning_methods import predict


class FreshValidationTests(unittest.TestCase):
    def test_exported_model_matches_fitted_pipeline(self):
        frame=pd.DataFrame(dict(x=[0.,1.,2.,3.,4.,5.],y=[0.,0.,1.,1.,2.,2.],fails_next_step=[0,0,0,1,1,1]))
        spec=fit_model('example',['x','y'],frame)
        pipe=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=42))
        pipe.fit(frame[['x','y']],frame.fails_next_step)
        restored=json.loads(json.dumps(spec))
        np.testing.assert_allclose(predict(restored,frame),pipe.predict_proba(frame[['x','y']])[:,1])

    def test_reservation_excludes_previous_candidates_and_observations(self):
        previous=Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                Path('splits').mkdir()
                Path('results').mkdir()
                Path('data/cifar-10-batches-py').mkdir(parents=True)
                Path('data/cifar-10-batches-py/test_batch').write_bytes(b'placeholder')
                old=[1000*c+i for c in range(10) for i in range(950)]
                pd.DataFrame(dict(image_id=old,source='test')).to_csv('splits/old.csv',index=False)
                Path('results/archive/historical_hybrids').mkdir(parents=True)
                pd.DataFrame(dict(image_id=[950],source=['test'])).to_csv('results/archive/historical_hybrids/old_audit.csv',index=False)
                root=Path('results/fresh'); root.mkdir()
                with patch('scripts.archive.fresh_trend_validation.pickle.load',return_value={b'labels':np.repeat(np.arange(10),1000)}):
                    manifest=reserve_fresh(root,count=100)
                d=pd.read_csv(manifest)
                self.assertEqual(len(d),100)
                self.assertTrue(d.class_id.value_counts().eq(10).all())
                self.assertFalse(set(d.image_id)&set(old+[950]))
                self.assertEqual(sha(manifest),json.loads((root/'reservation_audit.json').read_text())['manifest_sha256'])
            finally:
                os.chdir(previous)

    def test_complete_fresh_evaluation_never_fits_or_calibrates(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'frozen').mkdir()
            (root/'resnet50').mkdir()
            rows=[]
            for image in range(4):
                for severity in range(6):
                    for method in ['ig','gradcam']:
                        rows.append(dict(source='test',image_id=image,corruption='blur',severity=severity,
                            classifier='resnet50',true_label=0,clean_class=0,predicted_class=int(image%2==0 and severity>=3),
                            msp=.99-.03*severity,odin=.01*severity,method=method,valid_map=True,
                            loss=.1*severity,gain=.02*severity,weighted_wasserstein=.03*severity,
                            wasserstein=.04*severity,retention=1-.02*severity,
                            ig_steps=50,odin_temperature=1000,odin_epsilon=.0014))
            pd.DataFrame(rows).to_csv(root/'resnet50/images.csv',index=False)
            pd.DataFrame(dict(image_id=range(4),clean_correct=[1]*4)).to_csv(root/'resnet50/selection_audit.csv',index=False)
            manifest=root/'fresh_test_manifest.csv'
            pd.DataFrame(dict(image_id=range(4),source='test')).to_csv(manifest,index=False)
            learned=lambda name,col: dict(name=name,kind='learned',columns=[col],mean=[0.],scale=[1.],weights=[1.],intercept=0.,threshold=.53)
            artifact=dict(classifier='resnet50',settings=dict(ig_steps=50,odin_temperature=1000,odin_epsilon=.0014),methods=[
                dict(name='MSP',kind='msp',threshold=.08),dict(name='ODIN',kind='odin',threshold=.02),
                learned('Confidence + trend','unconfidence'),learned('Confidence + GRADCAM attribution/trends','gradcam_loss')])
            frozen=root/'frozen/resnet50.json'
            frozen.write_text(json.dumps(artifact))
            (root/'locked_protocol.json').write_text(json.dumps(dict(manifest_sha256=sha(manifest),artifact_sha256={'resnet50':sha(frozen)},primary_candidate='Confidence + GRADCAM attribution/trends',primary_control='MSP')))
            with patch('sklearn.preprocessing.StandardScaler.fit',side_effect=AssertionError('No fitting on fresh data')), patch('scripts.archive.fresh_trend_validation.calibrate_threshold',side_effect=AssertionError('No calibration on fresh data')):
                evaluate_fresh(root,'resnet50')
            self.assertTrue((root/'resnet50/evaluation/paired_effects.csv').exists())
            pairs=pd.read_csv(root/'resnet50/evaluation/paired_effects.csv')
            self.assertEqual(pairs.loc[pairs.primary,'control'].tolist(),['MSP'])

    def test_changed_protocol_rejected_before_opening_test_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'frozen').mkdir()
            (root/'frozen/resnet50.json').write_text('{}')
            (root/'fresh_test_manifest.csv').write_text('changed')
            (root/'locked_protocol.json').write_text(json.dumps(dict(manifest_sha256='wrong',artifact_sha256={'resnet50':'wrong'})))
            with self.assertRaisesRegex(ValueError,'Frozen protocol files changed'):
                evaluate_fresh(root,'resnet50')


if __name__=='__main__':
    unittest.main()
