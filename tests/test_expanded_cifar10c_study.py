import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from scripts.run_expanded_cifar10c_study import collect,recipes

class ExpandedCifarStudy(unittest.TestCase):
    def test_locked_cohorts(self):
        root=Path('results/expanded_cifar10c_5000dev_2000test')
        dev=pd.read_csv(root/'development_manifest.csv');test=pd.read_csv(root/'fresh_test_manifest.csv')
        audit=json.loads((root/'reservation/reservation_audit.json').read_text())
        self.assertEqual(len(dev),5000);self.assertEqual(len(test),2000)
        self.assertFalse(set(dev.image_id)&set(test.image_id))
        self.assertFalse(set(test.image_id)&set(audit['excluded_possible_prior_test_ids']))
        self.assertTrue((dev.groupby('class_id').size()==500).all())
        self.assertTrue((test.groupby('class_id').size()==200).all())
    def test_collect_uses_all_clean_correct_and_official_ig64(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.run_expanded_cifar10c_study.subprocess.run') as run:
                run.return_value.returncode=0
                collect(Path(folder),'resnet50','development',9999999999)
                args=run.call_args.args[0]
                self.assertIn('--all-clean-correct',args)
                self.assertEqual(args[args.index('--ig-steps')+1],'64')
                self.assertEqual(args[args.index('--dataset')+1],'cifar10c')
    def test_primary_recipes_unchanged(self):
        primary=[s for s in recipes() if s['primary']]
        self.assertEqual({s['name'] for s in primary},{'IG dynamic','GRADCAM dynamic'})
        self.assertTrue(all(len(s['columns'])==6 for s in primary))
