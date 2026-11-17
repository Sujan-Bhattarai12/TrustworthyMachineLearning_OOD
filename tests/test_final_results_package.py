import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT=Path('results/reporting/final')

class FinalResultsPackage(unittest.TestCase):
    def test_report_source_hashes(self):
        record=json.loads((ROOT/'provenance.json').read_text())
        for file,digest in record['data_sources'].items():
            self.assertEqual(hashlib.sha256(Path(file).read_bytes()).hexdigest(),digest,file)
        self.assertEqual(hashlib.sha256(Path('scripts/build_final_results.py').read_bytes()).hexdigest(),record['report_code_sha256'])
    def test_main_table_matches_original_counts(self):
        table=pd.read_csv(ROOT/'table2_main_comparison.csv');self.assertEqual(len(table),18)
        mapping={'ResNet-50':'resnet50','ViT-B/16':'vit_b_16','Swin-T':'swin_t'}
        for label,c in mapping.items():
            source=pd.read_csv(f'results/expanded_cifar10c_5000dev_2000test/evaluation/{c}/comparison.csv')
            for display,name in [('IG L/G/D + trends','IG dynamic'),('Grad-CAM L/G/D + trends','GRADCAM dynamic'),('MSP','MSP'),('ODIN','ODIN')]:
                a=table[(table.Classifier==label)&(table.Method==display)].iloc[0];b=source[source.model==name].iloc[0]
                self.assertAlmostEqual(a.Early_warning_pct,100*b.early_warning_rate)
                self.assertAlmostEqual(a.False_alarm_pct,100*b.false_alarm_rate)
    def test_cohort_counts_and_complete_ablation_table(self):
        table=pd.read_csv(ROOT/'table1_cohorts.csv');self.assertEqual(len(table),3)
        self.assertTrue((table.Valid_development_images==table.Fitting_images+table.Calibration_images).all())
        self.assertTrue((table.Valid_test_images==table.Failing_trajectories+table.Nonfailing_trajectories).all())
        ablations=pd.read_csv(ROOT/'table3_component_ablations.csv');self.assertEqual(len(ablations),36)
        self.assertTrue((ablations.groupby(['Classifier','Map']).size()==6).all())
    def test_four_nonempty_figures_and_three_tables(self):
        for i,name in enumerate(['warning_and_false_alarms','next_step_discrimination','attribution_ablations','corruption_generalization'],1):
            for ext in ['png','pdf']:self.assertGreater((ROOT/f'figure{i}_{name}.{ext}').stat().st_size,10000)
        for name in ['table1_cohorts','table2_main_comparison','table3_component_ablations']:
            for ext in ['csv','md','tex']:self.assertTrue((ROOT/f'{name}.{ext}').exists())
