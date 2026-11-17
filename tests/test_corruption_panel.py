import unittest
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_corruption_generalization import ROOT,PRIMARY,CORRUPTIONS,summarize

class FixedCorruptionPanel(unittest.TestCase):
    def test_locked_panel_and_stratified_subset(self):
        selected=pd.read_csv(ROOT/'manifest.csv');parent=pd.read_csv(PRIMARY/'fresh_test_manifest.csv')
        self.assertEqual(len(selected),200);self.assertTrue(set(selected.image_id)<=set(parent.image_id))
        self.assertTrue((selected.groupby('class_id').size()==20).all())
        self.assertEqual(len(CORRUPTIONS),4)
        self.assertFalse({'brightness','gaussian_noise','gaussian_blur'}&set(CORRUPTIONS))
    def test_warning_counts_before_first_failure(self):
        frame=pd.DataFrame(dict(source=['test']*12,image_id=[1]*6+[2]*6,corruption=['fog']*12,severity=list(range(6))*2,
            first_failure=[3.]*6+[np.nan]*6,eligible_fit=[False,True,True,False,False,False]*2,fails_next_step=[0,0,1,0,0,0]+[0]*6))
        score=np.array([0,0,.8,.9,.9,.9]+[0,.9,0,0,0,0])
        result=summarize(frame,score,.5)
        self.assertEqual(result['n_early'],1);self.assertEqual(result['n_false'],1)
        self.assertEqual(result['mean_lead_all_failures'],1.)
