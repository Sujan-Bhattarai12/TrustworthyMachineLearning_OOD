import unittest
from unittest.mock import Mock
import torch
from src.integrated_gradients import integrated_gradients
from scripts.collect_attribution_metrics import compute_ig

class IGStepTests(unittest.TestCase):
    def test_captum_default_and_legacy_step_forwarding(self):
        ig=Mock();x=torch.ones(1,3,2,2);ig.attribute.return_value=x
        compute_ig(ig,x,0)
        self.assertEqual(ig.attribute.call_args.kwargs['n_steps'],64)
        compute_ig(ig,x,0,steps=50)
        self.assertEqual(ig.attribute.call_args.kwargs['n_steps'],50)

    def test_custom_integrator_validates_configuration(self):
        for steps in [0,3,50,256]:
            with self.assertRaises(ValueError):integrated_gradients(None,None,0,steps=steps)
        class Linear(torch.nn.Module):
            def forward(self,x):return x.flatten(1).sum(1,keepdim=True)
        x=torch.ones(1,3,2,2)
        for steps in [1,2,4,8,16,32,64,128]:
            torch.testing.assert_close(integrated_gradients(Linear(),x,0,steps),x)
