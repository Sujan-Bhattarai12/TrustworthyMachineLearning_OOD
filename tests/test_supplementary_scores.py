import unittest
import math
import torch
from src.supplementary_scores import supplementary_scores

class SupplementaryScoreTests(unittest.TestCase):
    def test_uniform(self):
        result=supplementary_scores(torch.zeros(2,10))
        torch.testing.assert_close(result['Entropy'],torch.full((2,),math.log(10),dtype=torch.float64))
        torch.testing.assert_close(result['Energy'],-result['Entropy'])

    def test_logit_shift(self):
        logits=torch.tensor([[1.,2.,3.]])
        a=supplementary_scores(logits);b=supplementary_scores(logits+7)
        torch.testing.assert_close(a['Entropy'],b['Entropy'])
        torch.testing.assert_close(a['Energy']-7,b['Energy'])

    def test_extreme_and_temperature(self):
        result=supplementary_scores(torch.tensor([[10000.,-10000.]]),2)
        self.assertAlmostEqual(result['Entropy'].item(),0)
        self.assertAlmostEqual(result['Energy'].item(),-10000)
        with self.assertRaises(ValueError): supplementary_scores(torch.zeros(1,10),0)
        with self.assertRaises(ValueError): supplementary_scores(torch.tensor([[float('nan'),0.]]))
