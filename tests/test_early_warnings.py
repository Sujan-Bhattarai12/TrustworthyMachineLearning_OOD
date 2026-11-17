import unittest
import numpy as np
import pandas as pd
from scripts.evaluate_early_warnings import prepare, evaluate, calibrate_threshold, trajectory_outcomes


def dataset():
    rows=[]
    for image in range(30):
        for corruption in ['gaussian_noise','brightness']:
            for severity in range(6):
                failure = corruption == 'gaussian_noise' and image % 2 == 0 and severity == 3
                for method in ['ig','gradcam']:
                    rows.append(dict(source='train',image_id=image,corruption=corruption,severity=severity,
                                     true_label=0,clean_class=0,predicted_class=int(failure),
                                     msp=0.99-severity*0.03,odin=severity*0.03,method=method,valid_map=True,
                                     loss=severity*0.1,gain=severity*0.02,weighted_wasserstein=severity*0.04))
    return pd.DataFrame(rows)


class EarlyWarningTests(unittest.TestCase):
    def test_threshold_strict_and_conservative_for_ties(self):
        self.assertEqual(calibrate_threshold([1,1,2,2],0.25),2)
        self.assertEqual(calibrate_threshold([1,2,3,4],0.25),3)
        self.assertTrue(np.isnan(calibrate_threshold([],0.05)))

    def test_first_failure_stops_training_even_after_recovery(self):
        d,_=prepare(dataset())
        t=d[(d.image_id==0)&(d.corruption=='gaussian_noise')]
        self.assertTrue(t.first_failure.eq(3).all())
        self.assertEqual(t[t.eligible_fit].severity.tolist(),[1,2])

    def test_no_credit_for_alarm_at_failure_or_after(self):
        d,_=prepare(dataset())
        t=d[(d.image_id==0)&(d.corruption=='gaussian_noise')]
        result=trajectory_outcomes(t,(t.severity>=3).astype(float),0.5).iloc[0]
        self.assertFalse(result.early_warning)
        self.assertTrue(result.missed_warning)
        self.assertEqual(result.credited_lead_steps,0)
        early=trajectory_outcomes(t,(t.severity>=1).astype(float),0.5).iloc[0]
        self.assertEqual(early.credited_lead_steps,2)

    def test_invalid_component_excludes_whole_shared_trajectory(self):
        frame=dataset()
        frame.loc[(frame.image_id==0)&(frame.corruption=='brightness')&(frame.severity==2)&(frame.method=='ig'),'loss']=np.nan
        _,excluded=prepare(frame)
        self.assertEqual(len(excluded),1)

    def test_three_disjoint_roles_and_each_trajectory_once(self):
        d,_=prepare(dataset())
        summary,trajectories,thresholds,splits,paired=evaluate(d)
        self.assertTrue(splits.groupby(['fold','image_group']).role.nunique().eq(1).all())
        for _,group in splits.groupby('fold'):
            self.assertEqual(set(group.role),{'fit','calibration','evaluation'})
        self.assertFalse(trajectories.duplicated(['model','source','image_id','corruption']).any())
        self.assertTrue((thresholds.loc[thresholds.status=='ok','calibration_fpr']<=0.05).all())
        self.assertTrue({'MSP','ODIN','IG L raw','IG L+G fixed','GRADCAM L+G+D fixed'}<=set(summary.model))
        self.assertTrue((paired.n_earlier+paired.n_tied+paired.n_later==paired.n_shared_failures).all())


if __name__=='__main__':
    unittest.main()
