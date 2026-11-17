import unittest
import numpy as np
from test_early_warnings import dataset
from src.warning_features import validate_attribution_columns
from scripts.archive.fresh_trend_validation import fit_model
from scripts.freeze_warning_methods import predict
from src.attribution_features import add_features
from scripts.archive.test_ig_gradcam_fusion import fusion_scores


class AttributionOnlyTests(unittest.TestCase):
    def test_guard_rejects_confidence_and_labels(self):
        for col in ['unconfidence','msp','odin','delta_unconfidence','fails_next_step','first_failure','severity']:
            with self.assertRaises(ValueError):validate_attribution_columns(['ig_loss',col])

    def test_attribution_fit_rejects_hybrid(self):
        with self.assertRaises(ValueError):fit_model('IG attribution + trends',['ig_loss','unconfidence'],None)

    def test_prediction_checks_attribution_artifact_columns(self):
        with self.assertRaises(ValueError):predict(dict(role='attribution',kind='raw',columns=['msp']),None)

    def test_fusion_is_invariant_to_confidence_values(self):
        df=dataset();df['wasserstein']=df.severity*.03;df['retention']=1-df.severity*.04
        data,_=add_features(df)
        fit=data[data.image_id<10];cal=data[(data.image_id>=10)&(data.image_id<20)];test=data[data.image_id>=20]
        a=fusion_scores(fit,cal,test)
        changed=[]
        for d in [fit,cal,test]:
            d=d.copy();d['unconfidence']=np.linspace(.1,.9,len(d));d['delta_unconfidence']=np.linspace(.9,-.9,len(d));changed.append(d)
        b=fusion_scores(*changed)
        for name in ['IG+GradCAM fusion static','IG+GradCAM fusion + trends','IG separated + trends learned','GRADCAM separated + trends learned']:
            np.testing.assert_allclose(a[name][0],b[name][0]);np.testing.assert_allclose(a[name][1],b[name][1])
        self.assertFalse(any('MSP +' in name and ('IG' in name or 'GRADCAM' in name) for name in a))
        self.assertFalse(any(name.startswith('Confidence +') for name in a))


if __name__=='__main__':unittest.main()
