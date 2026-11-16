"""Changing micro grouping must preserve the existing eight-sample objective."""
import importlib
import unittest
import torch


class LossGroupingTests(unittest.TestCase):
    def helper(self):
        self.assertIsNotNone(importlib.util.find_spec('batch_loss'), 'batch_loss helper is missing')
        return importlib.import_module('batch_loss').token_mean_preserving_groups

    def test_partial_masks_preserve_loss_and_gradients_across_full_update(self):
        mean=self.helper()
        base=torch.arange(144*3*2,dtype=torch.float64).reshape(144,3,2)/31
        mask=torch.ones_like(base)
        mask[::3,1:]=0
        mask[8:16,1:]=0
        mask[24:32]=0  # Empty legacy groups contribute zero rather than NaN.
        reference=base.clone().requires_grad_()
        expected=sum((reference[i:i+8]*mask[i:i+8]).sum()/mask[i:i+8].sum().clamp_min(1)
            for i in range(0,144,8))/18
        expected.backward()
        for micro in (8,16,24):
            value=base.clone().requires_grad_()
            actual=sum(mean(value[i:i+micro],mask[i:i+micro])
                for i in range(0,144,micro))/(144//micro)
            actual.backward()
            torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(value.grad,reference.grad,rtol=1e-12,atol=1e-12)

    def test_masked_full_micro_mean_would_change_original_objective(self):
        mean=self.helper()
        value=torch.cat((torch.ones(8,2,1),torch.full((8,2,1),10.)))
        mask=torch.ones_like(value);mask[8:,1:]=0
        self.assertEqual(mean(value,mask).item(),5.5)
        self.assertEqual(((value*mask).sum()/mask.sum()).item(),4.0)

    def test_unmasked_mean_and_bad_geometry(self):
        mean=self.helper()
        value=torch.arange(48,dtype=torch.float32).reshape(24,2,1)
        torch.testing.assert_close(mean(value,None),value.mean())
        for size in (0,7,12,32):
            with self.assertRaises(ValueError):
                mean(torch.ones(size,2,1),None)
        with self.assertRaises(ValueError):
            mean(value,torch.ones(24,1,1))
        with self.assertRaises(ValueError):
            mean(value,None,base_micro=4)


if __name__ == '__main__':
    unittest.main()
